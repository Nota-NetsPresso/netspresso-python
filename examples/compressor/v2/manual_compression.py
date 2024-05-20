from netspresso import NetsPresso
from netspresso.enums import CompressionMethod, GroupPolicy, LayerNorm, Policy, StepOp
from netspresso.clients.compressor.v2.schemas.compression import Options

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare compressor
compressor = netspresso.compressor_v2()

# 2. Upload model
INPUT_MODEL_PATH = "./examples/sample_models/graphmodule.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]
model = compressor.upload_model(
    input_model_path=INPUT_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)

# 3. Select compression method
COMPRESSION_METHOD = CompressionMethod.PR_L2
compression_info = compressor.select_compression_method(
    model_id=model.ai_model_id,
    compression_method=COMPRESSION_METHOD,
)
print(compression_info)

# 4. Set params for compression(ratio or rank)
for available_layer in compression_info.available_layers[:5]:
    available_layer.values = [0.2]

# 5. Compress model
OUTPUT_DIR = "./outputs/compressed/graphmodule_manual"
compression_result = compressor.compress_model(
    compression=compression_info,
    output_dir=OUTPUT_DIR,
)

print(compression_result)