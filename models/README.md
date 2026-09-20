# Models Directory

Place your custom trained cement bag detection model here:
`models/best.pt`

### Model Requirements:
- Format: PyTorch YOLO format (`.pt`)
- Trained class: `cement_bag` (or set Target Class ID in sidebar configuration)
- If `models/best.pt` is absent, the application automatically falls back to `yolov8n.pt` for pipeline testing.
