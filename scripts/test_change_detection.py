from models.change_detection import get_change_detection_model


model = get_change_detection_model()

result = model.predict(
    image1="data/samples/test.jpg",
    image2="data/samples/test.tiff",
    output_path="outputs/change_map.png",
)

print("\nChange Detection Result")
print("=" * 40)

for key, value in result.items():
    print(f"{key}: {value}")