from enum import Enum


class QuantizationDataType(str, Enum):
    INT1 = "int1"
    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    INT5 = "int5"
    INT6 = "int6"
    INT7 = "int7"
    INT8 = "int8"
    INT9 = "int9"
    INT10 = "int10"
    INT11 = "int11"
    INT12 = "int12"
    INT13 = "int13"
    INT14 = "int14"
    INT15 = "int15"
    INT16 = "int16"
    INT17 = "int17"
    INT18 = "int18"
    INT19 = "int19"
    INT20 = "int20"
    INT21 = "int21"
    INT22 = "int22"
    INT23 = "int23"
    INT24 = "int24"
    INT25 = "int25"
    INT26 = "int26"
    INT27 = "int27"
    INT28 = "int28"
    INT29 = "int29"
    INT30 = "int30"
    INT31 = "int31"
    FLOAT32 = "float32" # When set to that value, excludes quantization from being quantized

    @classmethod
    def extract_number(cls, enum_value):
        return int(enum_value.lstrip("int"))


class QuantizationMode(str, Enum):
    PLAIN_QUANTIZATION = "plain_quantization"  # fixed_precision/quantization
    CUSTOM_QUANTIZATION = "custom_quantization"  # mixed_precision/quantization


class SimilarityMetric(str, Enum):
    SNR = "SNR"
