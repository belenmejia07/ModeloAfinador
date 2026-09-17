import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="Models/soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== INPUT ===")
for d in input_details:
    print(f"Nombre: {d['name']}")
    print(f"Shape: {d['shape']}")
    print(f"Tipo: {d['dtype']}")

print("\n=== OUTPUT ===")
for d in output_details:
    print(f"Nombre: {d['name']}")
    print(f"Shape: {d['shape']}")
    print(f"Tipo: {d['dtype']}")