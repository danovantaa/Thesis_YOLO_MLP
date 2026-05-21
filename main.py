from ultralytics import YOLO

model = YOLO("modelPreTrain\yolo11m.pt")
model.export(
    format="onnx",
    simplify=False,
    dynamic=False,
    opset=12
)
