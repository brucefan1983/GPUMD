from torchnep import train_nep

train_nep(
    config_file="nep.in",
    data_file="train.xyz",
    output_dir="output",
    print_interval=1,
    checkpoint_interval=50,
    use_compile=True,
)
