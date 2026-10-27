// Static import prevents dynamic-import thenable assimilation of OpenCV's namespace.
import * as cvModule from "@techstark/opencv-js";

export const openCv = cvModule.default ?? cvModule;
