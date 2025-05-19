from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
from scipy.spatial import KDTree

from gsMap.config import VisualizeConfig


def load_ldsc(ldsc_input_file):
    """
    Load LDSC results and classify based on significance threshold.
    p <= 0.05 -> "p ≤ 0.05" (red)
    p > 0.05 -> "p > 0.05" (blue)
    """

    ldsc = pd.read_csv(
        ldsc_input_file,
        compression="gzip",
        dtype={"spot": str, "p": float},
        index_col="spot",
        usecols=["spot", "p"],
    )
    ldsc["logp"] = -np.log10(ldsc["p"])
    logp_threshold = -np.log10(0.05)
    ldsc["significance"] = np.where(ldsc["logp"] >= logp_threshold, "p ≤ 0.05", "p > 0.05")
    return ldsc


# %%
def load_st_coord(adata, feature_series: pd.Series, annotation):
    spot_name = adata.obs_names.to_list()
    assert "spatial" in adata.obsm.keys(), "spatial coordinates are not found in adata.obsm"

    # to DataFrame
    space_coord = adata.obsm["spatial"]
    if isinstance(space_coord, np.ndarray):
        space_coord = pd.DataFrame(space_coord, columns=["sx", "sy"], index=spot_name)
    else:
        space_coord = pd.DataFrame(space_coord.values, columns=["sx", "sy"], index=spot_name)

    space_coord = space_coord[space_coord.index.isin(feature_series.index)]
    space_coord_concat = pd.concat([space_coord.loc[feature_series.index], feature_series], axis=1)
    space_coord_concat.head()
    if annotation is not None:
        annotation = pd.Series(
            adata.obs[annotation].values, index=adata.obs_names, name="annotation"
        )
        space_coord_concat = pd.concat([space_coord_concat, annotation], axis=1)
    return space_coord_concat


def estimate_point_size_for_plot(coordinates, DEFAULT_PIXEL_WIDTH=1000):
    tree = KDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    avg_min_distance = np.mean(distances[:, 1])
    # get the width and height of the plot
    width = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    height = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])

    scale_factor = DEFAULT_PIXEL_WIDTH / max(width, height)
    pixel_width = width * scale_factor
    pixel_height = height * scale_factor

    point_size = np.ceil(avg_min_distance * scale_factor)
    return (pixel_width, pixel_height), point_size


def draw_scatter(
    space_coord_concat,
    title=None,
    fig_style: Literal["dark", "light"] = "light",
    point_size: int = None,
    width=800,
    height=600,
    annotation=None,
    color_by="significance",
):
    # Set theme
    if fig_style == "dark":
        px.defaults.template = "plotly_dark"
    else:
        px.defaults.template = "plotly_white"

    # Categorical color map
    category_color_map = {
        "p ≤ 0.05": "red",
        "p > 0.05": "blue"
    }

    fig = px.scatter(
        space_coord_concat,
        x="sx",
        y="sy",
        color=color_by,
        color_discrete_map=category_color_map,
        symbol="annotation" if annotation is not None else None,
        title=title,
    )

    if point_size is not None:
        fig.update_traces(marker=dict(size=point_size, symbol="circle"))

    fig.update_layout(
        autosize=False,
        width=width,
        height=width,
        margin=dict(l=0, r=0, t=20, b=10),
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=1.0,
            font=dict(size=10),
        ),
        title=dict(
            y=0.98, x=0.5, xanchor="center", yanchor="top",
            font=dict(size=20)
        ),
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title=None, scaleanchor="y")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title=None)

    return fig


def run_Visualize(config: VisualizeConfig):
    print(f"------Loading LDSC results of {config.ldsc_save_dir}...")
    ldsc = load_ldsc(
        ldsc_input_file=Path(config.ldsc_save_dir)
        / f"{config.sample_name}_{config.trait_name}.csv.gz"
    )

    print(f"------Loading ST data of {config.sample_name}...")
    adata = sc.read_h5ad(f"{config.hdf5_with_latent_path}")

    space_coord_concat = load_st_coord(adata, ldsc, annotation=config.annotation)
    fig = draw_scatter(
        space_coord_concat,
        title=config.fig_title,
        fig_style=config.fig_style,
        point_size=config.point_size,
        width=config.fig_width,
        height=config.fig_height,
        annotation=config.annotation,
    )

    # Visualization
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    output_file_html = output_dir / f"{config.sample_name}_{config.trait_name}.html"
    output_file_pdf = output_dir / f"{config.sample_name}_{config.trait_name}.pdf"
    output_file_csv = output_dir / f"{config.sample_name}_{config.trait_name}.csv"

    fig.write_html(str(output_file_html))
    fig.write_image(str(output_file_pdf))
    space_coord_concat.to_csv(str(output_file_csv))

    print(
        f"------The visualization result is saved in a html file: {output_file_html} which can interactively viewed in a web browser and a pdf file: {output_file_pdf}."
    )
    print(f"------The visualization data is saved in a csv file: {output_file_csv}.")
