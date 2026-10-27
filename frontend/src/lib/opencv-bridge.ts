// Keep OpenCV behind a statically imported bridge: its module namespace may
// expose a then method that breaks direct dynamic-import promise assimilation.
import * as cvModule from "@techstark/opencv-js";

const moduleExports = cvModule as unknown as { default?: unknown };
export const openCv = moduleExports.default ?? cvModule;
