import matplotlib.pyplot as plt
from matchms import Spectrum
from matchms.plotting import plot_spectrum, plot_spectra_mirror
import torch

from .infer import process_input_data, process_model_output


def plot_sample_results(model, input_data, tokenizer, n_samples=5):
    has_labels = (
        "mzs" in input_data.columns and "intensities" in input_data.columns
    )

    # Process input data
    tokenized_data, supplementary_data, _ = process_input_data(input_data, tokenizer)

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

        # Add SMILES and other info as text on the plot
        smiles = input_data["smiles"][i]
        adduct = input_data["adduct"][i]
        collision_energy = input_data["collision_energy"][i]
        instrument_type = input_data["instrument_type"][i]

        # Plot predicted spectrum
        predicted_mzs = pred_mzs[i].numpy()
        predicted_intensities = pred_intensities[i].numpy()

        # Filter out zero intensities
        mask = predicted_intensities > 0
        predicted_mzs = predicted_mzs[mask]
        predicted_intensities = predicted_intensities[mask]

        # Sort by m/z, and sort predicted intensities by m/z
        sort_idx = predicted_mzs.argsort()
        predicted_mzs = predicted_mzs[sort_idx]
        predicted_intensities = predicted_intensities[sort_idx]

        predicted_spectrum = Spectrum(
            mz=predicted_mzs,
            intensities=predicted_intensities,
            metadata={
                "smiles": input_data["smiles"][i],
                "adduct": input_data["adduct"][i],
                "collision_energy": input_data["collision_energy"][i],
                "instrument_type": input_data["instrument_type"][i],
                "compound_name": "Predicted",
            }
        )

        # Plot actual spectrum
        if has_labels:
            actual_mzs = input_data["mzs"][i].to_numpy()
            actual_intensities = input_data["intensities"][i].to_numpy()
            # Sort by m/z, and sort predicted intensities by m/z
            sort_idx = actual_mzs.argsort()
            actual_mzs = actual_mzs[sort_idx]
            actual_intensities = actual_intensities[sort_idx]

            actual_spectrum = Spectrum(
                mz=actual_mzs,
                intensities=actual_intensities,
                metadata={
                    "smiles": input_data["smiles"][i],
                    "adduct": input_data["adduct"][i],
                    "collision_energy": input_data["collision_energy"][i],
                    "instrument_type": input_data["instrument_type"][i],
                    "compound_name": "Actual",
                }
            )   
            plot_spectra_mirror(
                spec_top=predicted_spectrum,
                spec_bottom=actual_spectrum,
            )
            plt.title(f"Predicted vs Actual Spectrum for Sample {i+1}")
        else:
            plot_spectrum(predicted_spectrum)
            plt.title(f"Predicted Spectrum for Sample {i+1}")

        plt.figtext(0.50, 0.92, f"SMILES: {smiles}", fontsize=8, ha='left', va="top")
        plt.figtext(0.50, 0.89, f"Adduct: {adduct}", fontsize=8, ha='left', va="top")
        plt.figtext(0.50, 0.86, f"Collision Energy: {collision_energy}", fontsize=8, ha='left', va="top")
        plt.figtext(0.50, 0.83, f"Instrument Type: {instrument_type}", fontsize=8, ha='left', va="top") 
        
        plt.xlabel("m/z")
        plt.ylabel("Intensity")

        plt.tight_layout()
        plt.show()
