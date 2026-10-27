import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pymupdf
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from app.documents.router import get_document_storage
from app.documents.storage import LocalFileStorage
from app.main import app
from app.worker.repository import IngestionRepository
from app.worker.runner import IngestionWorker

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    TEST_DATABASE_URL is None,
    reason="Set TEST_DATABASE_URL to run PostgreSQL integration tests.",
)


def test_project_crud_and_cursor_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TEST_DATABASE_URL is not None
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    created_ids: list[str] = []

    with TestClient(app) as client:
        try:
            for index in range(3):
                response = client.post(
                    "/api/v1/projects",
                    json={
                        "name": f"  Prototype project {uuid4()}  ",
                        "project_number": f"P-{index}",
                    },
                )
                assert response.status_code == 201
                assert response.json()["name"].startswith("Prototype project ")
                assert response.json()["created_by"] == "local-user"
                created_ids.append(response.json()["id"])

            first_page = client.get("/api/v1/projects?limit=2")
            assert first_page.status_code == 200
            first_page_data = first_page.json()
            assert len(first_page_data["items"]) == 2
            assert first_page_data["next_cursor"] is not None

            second_page = client.get(
                "/api/v1/projects",
                params={"limit": 2, "cursor": first_page_data["next_cursor"]},
            )
            assert second_page.status_code == 200
            assert len(second_page.json()["items"]) == 1
            assert second_page.json()["next_cursor"] is None

            project_id = created_ids[0]
            detail = client.get(f"/api/v1/projects/{project_id}")
            assert detail.status_code == 200

            updated = client.patch(
                f"/api/v1/projects/{project_id}",
                json={"description": "Updated project"},
            )
            assert updated.status_code == 200
            assert updated.json()["description"] == "Updated project"

            assert client.delete(f"/api/v1/projects/{project_id}").status_code == 204
            assert client.get(f"/api/v1/projects/{project_id}").status_code == 404

            assert (
                client.post(
                    "/api/v1/projects",
                    json={"name": "   "},
                ).status_code
                == 422
            )
            assert (
                client.get(
                    "/api/v1/projects",
                    params={"cursor": "invalid"},
                ).status_code
                == 422
            )
        finally:
            for project_id in created_ids:
                client.delete(f"/api/v1/projects/{project_id}")


def test_document_upload_persists_metadata_and_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert TEST_DATABASE_URL is not None
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    storage = LocalFileStorage(tmp_path / "uploads")
    app.dependency_overrides[get_document_storage] = lambda: storage
    project_id: str | None = None
    storage_key: str | None = None
    pdf_document = pymupdf.open()
    pdf_document.new_page().insert_text((72, 100), "Synthetic schedule text")
    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()

    with TestClient(app) as client:
        try:
            project = client.post(
                "/api/v1/projects",
                json={"name": f"Document integration {uuid4()}"},
            )
            assert project.status_code == 201
            project_id = project.json()["id"]

            upload = client.post(
                f"/api/v1/projects/{project_id}/documents",
                data={"role": "engineer", "document_type": "schedule"},
                files={
                    "file": (
                        "schedule.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload.status_code == 201
            assert upload.json()["status"] == "queued"
            assert upload.json()["extraction"]["status"] == "queued"
            assert upload.json()["file_name"] == "schedule.pdf"
            detail = client.get(f"/api/v1/documents/{upload.json()['id']}")
            assert detail.status_code == 200
            assert detail.json()["id"] == upload.json()["id"]

            with psycopg.connect(
                TEST_DATABASE_URL,
                autocommit=True,
                row_factory=dict_row,
            ) as connection:
                worker = IngestionWorker(
                    repository=IngestionRepository(connection),
                    storage=storage,
                    connection=connection,
                )
                assert worker.run_once() is True

            processed = client.get(f"/api/v1/documents/{upload.json()['id']}").json()
            assert processed["status"] == "ready"
            assert processed["extraction"]["status"] == "completed"
            assert processed["extraction"]["page_count"] == 1
            assert (
                processed["extraction"]["diagnostics"]["content_status"]
                == "selectable_text"
            )

            with psycopg.connect(TEST_DATABASE_URL, row_factory=dict_row) as verify:
                page = verify.execute(
                    """
                    SELECT text_content, page_width, page_height
                    FROM extracted_pages
                    WHERE extraction_run_id = (
                        SELECT id FROM extraction_runs WHERE document_id = %s
                    )
                    """,
                    (upload.json()["id"],),
                ).fetchone()
            assert "Synthetic schedule text" in page["text_content"]
            assert page["page_width"] > 0
            assert page["page_height"] > 0

            listing = client.get(f"/api/v1/projects/{project_id}/documents")
            assert listing.status_code == 200
            assert len(listing.json()["items"]) == 1
            document_id = listing.json()["items"][0]["id"]
            storage_key = f"{project_id}/{document_id}.pdf"
            assert (storage.root / storage_key).is_file()
        finally:
            if storage_key is not None:
                storage.delete(storage_key)
            if project_id is not None:
                client.delete(f"/api/v1/projects/{project_id}")
            app.dependency_overrides.clear()
