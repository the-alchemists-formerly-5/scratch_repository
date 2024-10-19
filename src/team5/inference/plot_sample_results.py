import matplotlib.pyplot as plt


def load_peft_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping parameter {name} due to shape mismatch")
        else:
            print(f"Ignoring unexpected key in state_dict: {name}")

    model.load_state_dict(model_state_dict, strict=False)


def plot_sample_results(model, input_data, tokenizer, n_samples=5):
    if "mzs" not in input_data.columns or "intensities" not in input_data.columns:
        print(
            "Input data does not contain 'mzs' and 'intensities' columns. Skipping plotting."
        )
        return

    # Process input data
    tokenized_data, supplementary_data = process_input_data(input_data, tokenizer)

    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_data["input_ids"],
            attention_mask=tokenized_data["attention_mask"],
            supplementary_data=supplementary_data,
        )

    # Process model output
    pred_mzs, pred_intensities = process_model_output(outputs)

    # Plot results for n_samples
    for i in range(min(n_samples, len(input_data))):
        plt.figure(figsize=(12, 6))

        # Plot actual spectrum
        actual_mzs = input_data["mzs"][i].to_numpy()
        actual_intensities = input_data["intensities"][i].to_numpy()
        plt.subplot(2, 1, 1)
        plt.stem(
            actual_mzs, actual_intensities, linefmt="b-", markerfmt="bo", basefmt=" "
        )
        plt.title(f"Actual Spectrum for Sample {i+1}")
        plt.xlabel("m/z")
        plt.ylabel("Intensity")

        # Plot predicted spectrum
        predicted_mzs = pred_mzs[i].numpy()
        predicted_intensities = pred_intensities[i].numpy()

        # Filter out zero intensities
        mask = predicted_intensities > 0
        predicted_mzs = predicted_mzs[mask]
        predicted_intensities = predicted_intensities[mask]

        plt.subplot(2, 1, 2)
        plt.stem(
            predicted_mzs,
            predicted_intensities,
            linefmt="r-",
            markerfmt="ro",
            basefmt=" ",
        )
        plt.title(f"Predicted Spectrum for Sample {i+1}")
        plt.xlabel("m/z")
        plt.ylabel("Intensity")

        # Add SMILES and other info as text on the plot
        smiles = input_data["smiles"][i]
        adduct = input_data["adduct"][i]
        collision_energy = input_data["collision_energy"][i]
        instrument_type = input_data["instrument_type"][i]

        plt.figtext(0.05, 0.5, f"SMILES: {smiles}", fontsize=8, va="center")
        plt.figtext(0.05, 0.47, f"Adduct: {adduct}", fontsize=8, va="center")
        plt.figtext(
            0.05, 0.44, f"Collision Energy: {collision_energy}", fontsize=8, va="center"
        )
        plt.figtext(
            0.05, 0.41, f"Instrument Type: {instrument_type}", fontsize=8, va="center"
        )

        plt.tight_layout()
        plt.show()
