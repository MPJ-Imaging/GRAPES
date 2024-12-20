##########################################################################################

# Graylevel Radial Analysis of ParticlES (GRAPES) Graphical User Interface (GUI)

##########################################################################################
# V0.0.1
# 2021-09-30
##########################################################################################
# Author: Matthew Jones
##########################################################################################
# Imports
##########################################################################################

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import pickle
import os
import tifffile
import numpy as np 
import GRAPES as grapes_module
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes
from skimage.measure import regionprops_table, mesh_surface_area
from skimage.morphology import ball, remove_small_holes
from scipy.ndimage import distance_transform_edt as dist_trans
from scipy.ndimage import convolve
from typing import Optional, Tuple, Any, Dict, Union, List
import logging
import warnings
import threading
import ast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##########################################################################################
# GRAPES Class
##########################################################################################

class GRAPES:
    def __init__(self):
        self.labels_arr = None
        self.grey_arr = None
        self.grapes_df = None
        self.radial_quartiles_df = None
        self.radial_deciles_df = None
        self.quartiles_df = None
        self.deciles_df = None
        self.grapes_properties = None
        self.additional_properties = None
        self.prop_image = None
        self.save_path = None

    def load_labels_image(self, filepath):
        """Load a multipage TIFF labels image using tifffile."""
        try:
            with tifffile.TiffFile(filepath) as tif:
                images = tif.asarray()
            print(f"Loaded labels image with shape: {images.shape}")
            self.labels_arr = images
            return images
        except Exception as e:
            print(f"Error loading labels image: {e}")
            raise e

    def load_gray_level_image(self, filepath):
        """Load a multipage TIFF gray level image using tifffile."""
        try:
            with tifffile.TiffFile(filepath) as tif:
                images = tif.asarray()
            print(f"Loaded gray level image with shape: {images.shape}")
            self.grey_arr = images
            return images
        except Exception as e:
            print(f"Error loading gray level image: {e}")
            raise e

    def load_grapes_dataframe(self, filepath):
        """Load a pre-calculated GRAPES dataframe from a pickle file."""
        try:
            with open(filepath, 'rb') as f:
                df = pickle.load(f)
            print(f"Loaded GRAPES DataFrame with {len(df)} records")
            self.grapes_df = df
            return df
        except Exception as e:
            print(f"Error loading GRAPES DataFrame: {e}")
            raise e

    def save_dataframes(self,
                        output_dir: str,
                        formats: Union[str, List[str]],
                        excel_filename: Optional[str] = "dataframes.xlsx",
                        separate_excel_sheets: bool = True,
                        overwrite: bool = False,
                        verbose: bool = True) -> str:
        """
        Wrapper method to save DataFrames using the grapes_module's save_dataframes function.

        Args:
            output_dir (str): Directory to save the DataFrames.
            formats (Union[str, List[str]]): Formats to save ('xlsx', 'pkl').
            excel_filename (Optional[str], optional): Filename for Excel. Defaults to "dataframes.xlsx".
            separate_excel_sheets (bool, optional): Whether to save each DataFrame in separate sheets. Defaults to True.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            verbose (bool, optional): Whether to print verbose messages. Defaults to True.

        Returns:
            str: Result message indicating success or failure.
        """
        # Check if at least one DataFrame exists
        dataframes = {}
        if self.grapes_df is not None:
            dataframes["grapes_df"] = self.grapes_df
        if self.radial_quartiles_df is not None:
            dataframes["radial_quartiles_df"] = self.radial_quartiles_df
        if self.radial_deciles_df is not None:
            dataframes["radial_deciles_df"] = self.radial_deciles_df
        if self.quartiles_df is not None:
            dataframes["quartiles_df"] = self.quartiles_df
        if self.deciles_df is not None:
            dataframes["deciles_df"] = self.deciles_df

        if not dataframes:
            return "Error: No DataFrames available to save. Perform analyses first."

        try:
            grapes_module.save_dataframes(
                dataframes=dataframes,
                output_dir=output_dir,
                formats=formats,
                excel_filename=excel_filename,
                separate_excel_sheets=separate_excel_sheets,
                overwrite=overwrite,
                verbose=verbose
            )
            saved_dfs = ", ".join(dataframes.keys())
            return f"DataFrame(s) successfully saved: {saved_dfs} to '{output_dir}'."
        except Exception as e:
            return f"Error saving DataFrame(s): {e}"

    # Analysis Functions
    def calc_GRAPES(
        self,
        normalised_by, 
        start_at, 
        pixel_size, 
        fill_label_holes, 
        min_hole_size,
        anisotropy,
        parallel,
        n_jobs
        ):
        """Perform GRAPES analysis with the given parameters."""
        if self.labels_arr is None or self.grey_arr is None:
            return "Error: Labels Image and Gray Level Image must be loaded first."
        try:
            grapes_df = grapes_module.GRAPES(
                labels_arr=self.labels_arr,
                grey_arr=self.grey_arr,
                normalised_by=normalised_by,
                start_at=start_at,
                pixel_size=pixel_size,
                fill_label_holes=fill_label_holes,
                min_hole_size=min_hole_size,
                anisotropy=anisotropy,
                order='C',
                black_border=True,
                parallel=parallel,
                n_jobs=n_jobs
            )
            self.grapes_df = grapes_df
            return f"GRAPES Analysis Completed. DataFrame with {len(grapes_df)} records created."
        except Exception as e:
            return f"Error during GRAPES analysis: {e}"

    def calc_radial_layers_quartiles(self, prop):
        """Calculate quartiles of radial properties"""
        if self.grapes_df is None:
            return "Error: GRAPES DataFrame must be loaded/calculated first."
        try:
            radial_quartiles_df = grapes_module.radial_layers_quartiles(
                grapes_df=self.grapes_df,
                prop=prop,
                grapes_properties=None
            )
            self.radial_quartiles_df = radial_quartiles_df
            return f"Analysis executed. DataFrame with {len(radial_quartiles_df)} records created."
        except Exception as e:
            return f"Error during Quartiles Analysis: {e}"

    def calc_radial_layers_deciles(self, prop):
        """Calculate deciles of radial properties"""
        if self.grapes_df is None:
            return "Error: GRAPES DataFrame must be loaded/calculated first."
        try:
            radial_deciles_df = grapes_module.radial_layers_deciles(
                grapes_df=self.grapes_df,
                prop=prop,
                grapes_properties=None
            )
            self.radial_deciles_df = radial_deciles_df
            return f"Analysis executed. DataFrame with {len(radial_deciles_df)} records created."
        except Exception as e:
            return f"Error during Deciles Analysis: {e}"

    def calc_quartiles(self, prop):
        """Calculate quartiles of non-radial properties"""
        if self.grapes_df is None:
            return "Error: GRAPES DataFrame must be loaded/calculated first."
        try:
            quartiles_df = grapes_module.compute_quartile_statistics(
                grapes_df=self.grapes_df,
                prop=prop,
                additional_properties=None
            )
            self.quartiles_df = quartiles_df
            return f"Analysis executed. DataFrame with {len(quartiles_df)} records created."
        except Exception as e:
            return f"Error during Quartiles Analysis: {e}"

    def calc_deciles(self, prop):
        """Calculate deciles of non-radial properties"""
        if self.grapes_df is None:
            return "Error: GRAPES DataFrame must be loaded/calculated first."
        try:
            deciles_df = grapes_module.compute_decile_statistics(
                grapes_df=self.grapes_df,
                prop=prop,
                additional_properties=None
            )
            self.deciles_df = deciles_df
            return f"Analysis executed. DataFrame with {len(deciles_df)} records created."
        except Exception as e:
            return f"Error during Deciles Analysis: {e}"

    def prop_2_image(self, prop):
        """Assign Property values to labels in image space"""
        if self.grapes_df is None:
            return "Error: GRAPES DataFrame must be loaded/calculated first."
        elif self.labels_arr is None:
            return "Error: Labels Image must be loaded first."
        try:
            prop_img = grapes_module.prop_2_image(
                labels=self.labels_arr,
                df=self.grapes_df,
                prop=prop,
                save_path=None,
                show=False
            )
            self.prop_image = prop_img
            return f"Property Image created with shape: {prop_img.shape}" 
        except Exception as e:
            return f"Error creating Property Image: {e}"       

    # Plotting Functions
    def plot_particle_image_method(
        self,
        label: int,
        image_type: str = 'distance_transform',
        slice_idx: Optional[int] = None,
        title_suffix: Optional[str] = None,
        xlabel: str = 'X-axis',
        ylabel: str = 'Y-axis',
        figsize: tuple = (8, 6),
        cmap: str = 'viridis',
        colorbar: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Figure]:
        """
        Wrapper method for the plot_particle_image function from the grapes module.

        Args:
            label (int): The label of the particle to plot.
            image_type (str, optional): Type of image to plot. Defaults to 'distance_transform'.
            slice_idx (int, optional): The slice index to plot. Defaults to None.
            title_suffix (str, optional): Additional string to append to the plot title. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to 'X-axis'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Y-axis'.
            figsize (tuple, optional): Size of the figure in inches. Defaults to (8, 6).
            cmap (str, optional): Colormap to use for the image. Defaults to 'viridis'.
            colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
            save_path (str, optional): File path to save the plot image. Defaults to None.

        Returns:
            Optional[Figure]: The Matplotlib Figure object if the plot is generated, else None.
        """
        if self.grapes_df is None:
            return "Error: GRAPES DataFrame must be loaded/calculated first."
        try:
            fig = grapes_module.plot_particle_image(
                df=self.grapes_df,
                label=label,
                image_type=image_type,
                slice_idx=slice_idx,
                title_suffix=title_suffix,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                cmap=cmap,
                colorbar=colorbar,
                save_path=save_path,
                show = True 
            )
            return fig
        except Exception as e:
            warnings.warn(f"Failed to plot particle image: {e}")
            return None

    def plot_quartile_radial_layers_method(
        self,
        prop: str = 'radial_layers',  # Changed from 'property' to 'prop'
        error_type: str = 'SE',
        title: Optional[str] = None,
        xlabel: str = 'Radial Layer',
        ylabel: str = 'Mean Value',
        figsize: tuple = (10, 6),
        palette: Optional[List[str]] = None
    ) -> Optional[plt.Figure]:
        """
        Wrapper method for the plot_quartile_radial_layers function from the grapes module.

        Args:
            prop (str, optional): GRAPES property to plot. Defaults to 'radial_layers'.
            error_type (str, optional): Type of error bars to use ('SE' or 'Std'). Defaults to 'SE'.
            title (str, optional): Title of the plot. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Radial Layer'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Mean Value'.
            figsize (tuple, optional): Size of the figure. Defaults to (10, 6).
            palette (List[str], optional): List of colors for quartiles. Defaults to None.

        Returns:
            Optional[plt.Figure]: The Matplotlib Figure object if the plot is generated, else None.
        """
        try:
            fig = grapes_module.plot_quartile_radial_layers(
                quartiles_df=self.radial_quartiles_df,
                prop=prop,
                error_type=error_type,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                palette=palette,
                show=True  # This will display the plot in a new window
            )
            return fig
        except Exception as e:
            warnings.warn(f"Failed to plot quartile radial layers: {e}")
            return None

    def plot_decile_radial_layers_method(
        self,
        prop: str = 'radial_layers',  # Changed from 'property' to 'prop'
        error_type: str = 'SE',
        title: Optional[str] = None,
        xlabel: str = 'Radial Layer',
        ylabel: str = 'Mean Value',
        figsize: tuple = (10, 6),
        palette: Optional[List[str]] = None
    ) -> Optional[plt.Figure]:
        """
        Wrapper method for the plot_decile_radial_layers function from the grapes module.

        Args:
            prop (str, optional): GRAPES property to plot. Defaults to 'radial_layers'.
            error_type (str, optional): Type of error bars to use ('SE' or 'Std'). Defaults to 'SE'.
            title (str, optional): Title of the plot. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Radial Layer'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Mean Value'.
            figsize (tuple, optional): Size of the figure. Defaults to (10, 6).
            palette (List[str], optional): List of colors for deciles. Defaults to None.

        Returns:
            Optional[plt.Figure]: The Matplotlib Figure object if the plot is generated, else None.
        """
        try:
            fig = grapes_module.plot_decile_radial_layers(
                deciles_df=self.radial_deciles_df,
                prop=prop,
                error_type=error_type,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                palette=palette,
                show=True  # This will display the plot in a new window
            )
            return fig
        except Exception as e:
            warnings.warn(f"Failed to plot decile radial layers: {e}")
            return None

    
    def plot_radial_intensities_method(
        self,
        label: int,
        plotting: str = 'normalised',
        title_suffix: Optional[str] = None,
        xlabel: str = 'Radial Position',
        ylabel: Optional[str] = None,
        figsize: tuple = (8, 6),
        color: str = 'blue',
        marker: Union[str, None] = 'o',
        linestyle: str = '-',
        linewidth: float = 1.5
    ) -> Optional[Figure]:
        """
        Wrapper method for the plot_radial_intensities function from the grapes module.

        Args:
            label (int): The label of the particle to plot.
            plotting (str, optional): Type of plot - 'normalised' or 'graylevels'. 
                                      Defaults to 'normalised'.
            title_suffix (str, optional): Additional string to append to the plot title.
                                          Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Radial Position'.
            ylabel (str, optional): Label for the y-axis. Defaults based on plotting type.
            figsize (tuple, optional): Size of the figure in inches. Defaults to (8, 6).
            color (str, optional): Color of the plot line. Defaults to 'blue'.
            marker (str or None, optional): Marker style for the plot. Set to None for no markers.
                                             Defaults to 'o'.
            linestyle (str, optional): Line style for the plot. Defaults to '-'.
            linewidth (float, optional): Width of the plot line. Defaults to 1.5.

        Returns:
            Optional[Figure]: The Matplotlib Figure object if the plot is generated, else None.
        """
        try:
            fig = grapes_module.plot_radial_intensities(
                df=self.grapes_df,
                label=label,
                plotting=plotting,
                title_suffix=title_suffix,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                show=True  # This will display the plot in a new window
                # save_path parameter removed 
            )
            return fig
        except Exception as e:
            warnings.warn(f"Failed to plot radial intensities: {e}")
            return None

    def plot_vol_slice_images_method(
        self,
        im: np.ndarray,
        slice_idx: Optional[int] = None,
        title_suffix: Optional[str] = None,
        xlabel: str = 'X-axis',
        ylabel: str = 'Y-axis',
        figsize: tuple = (8, 6),
        cmap: str = 'viridis',
        colorbar: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Figure]:
        """ Wrapper around a simple matplotlib.pyplot imshow function to plot images. """
        try:
            fig = grapes_module.plot_vol_slice_images(
                im = im,
                slice_idx=slice_idx,
                title_suffix=title_suffix,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                cmap=cmap,
                colorbar=colorbar,
                save_path=save_path,
                show=True
            )
            return fig
        except Exception as e:
            warnings.warn(f"Failed to plot images: {e}")
            return None

##########################################################################################
# GRAPES GUI
##########################################################################################

class GrapesGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GRAPES GUI V0.01")
        self.geometry("800x500")
        self.grapes = GRAPES()

        # Set Window Icon
        self.set_window_icon()

        # Initialize Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both')

        # Initialize Tabs
        self.home_tab = HomeTab(self.notebook, self.grapes)
        self.io_tab = IOTab(self.notebook, self.grapes)
        self.analysis_tab = AnalysisTab(self.notebook, self.grapes)
        self.plotting_tab = PlottingTab(self.notebook, self.grapes)

        # Add Tabs to Notebook
        self.notebook.add(self.home_tab, text='Home')
        self.notebook.add(self.io_tab, text='I/O')
        self.notebook.add(self.analysis_tab, text='Analysis')
        self.notebook.add(self.plotting_tab, text='Plotting')

    def set_window_icon(self):
        try:
            icon_path = os.path.join('assets', 'icon.png')
            icon_image = Image.open(icon_path)
            # Recommended size: 32x32 pixels
            icon_image = icon_image.resize((32, 32), Image.ANTIALIAS)
            self.icon = ImageTk.PhotoImage(icon_image)
            self.iconphoto(False, self.icon)
        except Exception as e:
            print(f"Error loading window icon: {e}")
            # Optionally, set a default icon or leave it as default

##########################################################################################
# Home Tab
##########################################################################################

class HomeTab(ttk.Frame):
    def __init__(self, parent, grapes):
        super().__init__(parent)
        self.grapes = grapes

        # Title Label
        welcome_text = "Welcome to GRAPES GUI - Radial Analysis of Particle Graylevels for Microscopy"
        self.title_label = ttk.Label(
            self,
            text=welcome_text,
            wraplength=600,
            font=("Arial", 20, "bold"),
            anchor='center',
            justify='center'
        )
        self.title_label.pack(pady=20)

        # Logo Image
        try:
            logo_image = Image.open(os.path.join('assets', 'logo.png'))
            logo_image = logo_image.resize((200, 200), Image.ANTIALIAS)
            self.logo = ImageTk.PhotoImage(logo_image)
            self.logo_label = ttk.Label(self, image=self.logo)
            self.logo_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading logo image: {e}")
            self.logo_label = ttk.Label(self, text="[Logo Image]")
            self.logo_label.pack(pady=10)

        # Introduction Textbox
        intro_text = (
            "GRAPES GUI provides a user-friendly interface for performing Radial Analysis of Particle Gray levels in Microscopy images. "
            "Navigate through the tabs to load your data, perform various analyses, and visualize the results through customizable plots."
        )
        self.intro_label = ttk.Label(
            self,
            text=intro_text,
            wraplength=700,
            font=("Arial", 12),
            justify='center'
        )
        self.intro_label.pack(pady=20)

##########################################################################################
# IO Tab
##########################################################################################

class IOTab(ttk.Frame):
    def __init__(self, parent, grapes):
        super().__init__(parent)
        self.grapes = grapes

        # Dictionary to store references to Entry widgets
        self.entries = {}

        # Labels and Entries for Loading Files
        self.create_loading_widget("Load Labels Image (single or multipage .tif or .tiff)", "labels_image", self.load_labels_image)
        self.create_loading_widget("Load Gray Level Image (single or multipage .tif or .tiff)", "gray_level_image", self.load_gray_level_image)
        self.create_loading_widget("Load pre-calculated GRAPES dataframe (.pkl)", "grapes_df", self.load_grapes_dataframe)
        
        # Add Save Button for Saving DataFrames
        self.create_save_widget("Save GRAPES DataFrame(s)", self.save_dataframes)

    def create_loading_widget(self, label_text, entry_key, command):
        frame = ttk.Frame(self)
        frame.pack(pady=10, padx=20, fill='x')

        label = ttk.Label(frame, text=label_text)
        label.pack(side='left', padx=(0,10))

        entry = ttk.Entry(frame, width=50)
        entry.pack(side='left', padx=(0,10))

        browse_button = ttk.Button(frame, text="Browse", command=lambda: command(entry))
        browse_button.pack(side='left')

        # Store the Entry reference
        self.entries[entry_key] = entry

    def create_save_widget(self, label_text, command):
        frame = ttk.Frame(self)
        frame.pack(pady=20, padx=20, fill='x')

        label = ttk.Label(frame, text=label_text)
        label.pack(side='left', padx=(0,10))

        save_button = ttk.Button(frame, text="Save", command=command)
        save_button.pack(side='left')

    def load_labels_image(self, entry):
        filepath = filedialog.askopenfilename(
            title="Select Labels Image",
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)
            # Load the labels image using tifffile
            try:
                images = self.grapes.load_labels_image(filepath)
                messagebox.showinfo("Load Labels Image", f"Labels Image Loaded with shape: {images.shape}")
            except Exception as e:
                messagebox.showerror("Load Labels Image", f"Failed to load Labels Image: {e}")

    def load_gray_level_image(self, entry):
        filepath = filedialog.askopenfilename(
            title="Select Gray Level Image",
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)
            # Load the gray level image using tifffile
            try:
                images = self.grapes.load_gray_level_image(filepath)
                messagebox.showinfo("Load Gray Level Image", f"Gray Level Image Loaded with shape: {images.shape}")
            except Exception as e:
                messagebox.showerror("Load Gray Level Image", f"Failed to load Gray Level Image: {e}")

    def load_grapes_dataframe(self, entry):
        filepath = filedialog.askopenfilename(
            title="Select GRAPES DataFrame",
            filetypes=[("Pickle files", "*.pkl")]
        )
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)
            # Load the GRAPES dataframe
            try:
                result = self.grapes.load_grapes_dataframe(filepath)
                messagebox.showinfo("Load GRAPES DataFrame", f"DataFrame Loaded with {len(result)} records")
            except Exception as e:
                messagebox.showerror("Load GRAPES DataFrame", f"Failed to load DataFrame: {e}")

    def save_dataframes(self):
        """Open a window to specify save options and execute saving DataFrames."""
        # Open a new window for save options
        save_window = tk.Toplevel(self)
        save_window.title("Save GRAPES DataFrame(s)")
        save_window.geometry("500x600")

        # Output Directory Selection
        ttk.Label(save_window, text="Output Directory:").pack(pady=10)
        output_dir_entry = ttk.Entry(save_window, width=50)
        output_dir_entry.pack(pady=5)
        browse_output_dir_button = ttk.Button(save_window, text="Browse", command=lambda: self.browse_directory(save_window, output_dir_entry))
        browse_output_dir_button.pack(pady=5)

        # Store the output_dir_entry in self.entries
        self.entries["output_dir"] = output_dir_entry

        # Formats Selection (Checkboxes for 'xlsx' and 'pkl')
        ttk.Label(save_window, text="Select Save Formats:").pack(pady=10)
        formats_frame = ttk.Frame(save_window)
        formats_frame.pack(pady=5)

        self.xlsx_var = tk.BooleanVar(value=True)
        self.pkl_var = tk.BooleanVar(value=True)

        xlsx_check = ttk.Checkbutton(formats_frame, text="Excel (.xlsx)", variable=self.xlsx_var)
        xlsx_check.pack(side='left', padx=10)
        pkl_check = ttk.Checkbutton(formats_frame, text="Pickle (.pkl)", variable=self.pkl_var)
        pkl_check.pack(side='left', padx=10)

        # Excel Filename Entry (only enabled if 'xlsx' is selected)
        self.excel_filename_label = ttk.Label(save_window, text="Excel Filename:")
        self.excel_filename_label.pack(pady=10)
        self.excel_filename_entry = ttk.Entry(save_window, width=30)
        self.excel_filename_entry.pack(pady=5)
        self.excel_filename_entry.insert(0, "dataframes.xlsx")

        # Separate Excel Sheets Checkbox (only enabled if 'xlsx' is selected)
        self.separate_sheets_var = tk.BooleanVar(value=True)
        self.separate_sheets_check = ttk.Checkbutton(
            save_window,
            text="Save each DataFrame in separate Excel sheets",
            variable=self.separate_sheets_var
        )
        self.separate_sheets_check.pack(pady=10)

        # Overwrite Checkbox
        self.overwrite_var = tk.BooleanVar(value=False)
        overwrite_check = ttk.Checkbutton(
            save_window,
            text="Overwrite existing files",
            variable=self.overwrite_var
        )
        overwrite_check.pack(pady=10)

        # Execute Save Button
        execute_save_button = ttk.Button(save_window, text="Save DataFrame(s)", command=lambda: self.execute_save(save_window))
        execute_save_button.pack(pady=20)

        # Disable Excel-related widgets if 'xlsx' is not selected
        self.xlsx_var.trace('w', self.toggle_excel_widgets)
        self.toggle_excel_widgets()

        # Docstring Display
        ttk.Label(save_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(save_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        docstring = grapes_module.save_dataframes.__doc__ or "No documentation available."
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def browse_directory(self, window, entry_widget):
        """Open a directory selection dialog and update the entry widget."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)

    def toggle_excel_widgets(self, *args):
        """Enable or disable Excel-related widgets based on 'xlsx' checkbox."""
        if self.xlsx_var.get():
            self.excel_filename_label.config(state='normal')
            self.excel_filename_entry.config(state='normal')
            self.separate_sheets_check.config(state='normal')
        else:
            self.excel_filename_label.config(state='disabled')
            self.excel_filename_entry.config(state='disabled')
            self.separate_sheets_check.config(state='disabled')

    def execute_save(self, window):
        """Gather user inputs and execute the save_dataframes function."""
        # Get Output Directory
        output_dir_entry = self.entries.get("output_dir")
        if output_dir_entry:
            output_dir = output_dir_entry.get().strip()
        else:
            output_dir = ""

        if not output_dir:
            messagebox.showerror("Input Error", "Please specify an output directory.")
            return

        # Get Formats
        formats = []
        if self.xlsx_var.get():
            formats.append('xlsx')
        if self.pkl_var.get():
            formats.append('pkl')
        if not formats:
            messagebox.showerror("Input Error", "Please select at least one format to save.")
            return

        # Get Excel Filename
        excel_filename = self.excel_filename_entry.get().strip()
        if 'xlsx' in formats and not excel_filename.endswith('.xlsx'):
            messagebox.showerror("Input Error", "Excel filename must end with '.xlsx'.")
            return

        # Get Separate Excel Sheets option
        separate_excel_sheets = self.separate_sheets_var.get() if 'xlsx' in formats else False

        # Get Overwrite option
        overwrite = self.overwrite_var.get()

        # Prepare formats parameter
        if len(formats) == 1:
            formats_param = formats[0]
        else:
            formats_param = formats

        # Execute the save in a separate thread to keep GUI responsive
        import threading

        def save_thread():
            result = self.grapes.save_dataframes(
                output_dir=output_dir,
                formats=formats_param,
                excel_filename=excel_filename,
                separate_excel_sheets=separate_excel_sheets,
                overwrite=overwrite,
                verbose=True
            )
            # Update the GUI in the main thread
            self.after(0, lambda: self.display_save_result(result, window))

        threading.Thread(target=save_thread).start()

    def display_save_result(self, result, window):
        """Display the result of the save operation."""
        if "successfully saved" in result.lower():
            messagebox.showinfo("Save Successful", result)
        else:
            messagebox.showerror("Save Error", result)
        window.destroy()

##########################################################################################
# Analysis Tab
##########################################################################################

class AnalysisTab(ttk.Frame):
    def __init__(self, parent, grapes):
        super().__init__(parent)
        self.grapes = grapes

        # Example analysis functions
        self.analysis_functions = {
            "Analyse Images with GREAT2 Algorithm": self.calc_GRAPES,
            "Radial Layer Quartiles": self.calc_radial_layers_quartiles,
            "Radial Layer Deciles": self.calc_radial_layers_deciles,
            "Quartiles (Non-radial)": self.calc_quartiles,
            "Deciles (Non-radial)": self.calc_deciles,
            "Assign Property to Labels in Image Space": self.prop_2_image
            # Add more functions here as needed
        }

        # Create Grid of Buttons (2x3)
        self.create_buttons_grid()

    def create_buttons_grid(self):
        functions = list(self.analysis_functions.keys())
        rows = 3
        cols = 2
        for index, func_name in enumerate(functions):
            row = index // cols
            col = index % cols
            button = ttk.Button(self, text=func_name, command=self.analysis_functions[func_name])
            button.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')

        # Configure grid weights for responsiveness
        for i in range(rows):
            self.grid_rowconfigure(i, weight=1)
        for j in range(cols):
            self.grid_columnconfigure(j, weight=1)

    def calc_GRAPES(self):
        self.open_grapes_window("Analyse Images with GREAT2 Algorithm")

    def calc_radial_layers_quartiles(self):
        self.open_radial_quartiles_window("Radial Layer Property Quartiles")

    def calc_radial_layers_deciles(self):
        self.open_radial_deciles_window("Radial Layer Property Deciles")

    def calc_quartiles(self):
        self.open_quartiles_window("Property Quartiles")

    def calc_deciles(self):
        self.open_deciles_window("Property Deciles")

    def prop_2_image(self):
        self.open_prop_2_image_window("Assign Property to Labels in Image Space")

    def open_grapes_window(self, title):
        """Open a window to perform GRAPES analysis with specified parameters."""
        # Open a new window to enter arguments and display docstring
        args_window = tk.Toplevel(self)
        args_window.title(title)
        args_window.geometry("500x800")

        # Fetch the docstring
        docstring = grapes_module.GRAPES.__doc__ or "No documentation available."

        # Parameter Inputs

        # 1. normalised_by Dropdown
        ttk.Label(args_window, text="Normalization Method:").pack(pady=10)
        normalised_by_var = tk.StringVar()
        normalised_by_combo = ttk.Combobox(args_window, textvariable=normalised_by_var)
        normalised_by_combo['values'] = ("radial_max", "surface")
        normalised_by_combo.current(0)
        normalised_by_combo.pack(pady=5)

        # 2. start_at Dropdown
        ttk.Label(args_window, text="Start At:").pack(pady=10)
        start_at_var = tk.StringVar()
        start_at_combo = ttk.Combobox(args_window, textvariable=start_at_var)
        start_at_combo['values'] = ("edge", "center")
        start_at_combo.current(0)
        start_at_combo.pack(pady=5)

        # 3. pixel_size Entry
        ttk.Label(args_window, text="Pixel Size (optional):").pack(pady=10)
        pixel_size_entry = ttk.Entry(args_window, width=30)
        pixel_size_entry.pack(pady=5)

        # 4. fill_label_holes Checkbox
        fill_label_holes_var = tk.BooleanVar()
        fill_label_holes_check = ttk.Checkbutton(
            args_window,
            text="Fill Label Holes",
            variable=fill_label_holes_var
        )
        fill_label_holes_check.pack(pady=10)

        # 5. min_hole_size Entry
        ttk.Label(args_window, text="Minimum Hole Size:").pack(pady=10)
        min_hole_size_entry = ttk.Entry(args_window, width=30)
        min_hole_size_entry.pack(pady=5)
        min_hole_size_entry.insert(0, "5000")  # Default value

        # 6. Anisotropy text entry
        ttk.Label(args_window, text="Anisotropy:").pack(pady=10)
        anisotropy_entry = ttk.Entry(args_window, width=30)
        anisotropy_entry.pack(pady=5)
        anisotropy_entry.insert(0, "(1.0, 1.0, 1.0)")  # Default value

        # 7. parallel text entry
        ttk.Label(args_window, text="Parallel (threads): Enter +'ve integer").pack(pady=10)
        parallel_entry = ttk.Entry(args_window, width=30)
        parallel_entry.pack(pady=5)
        parallel_entry.insert(0, "4")  # Default value

        # 8. n_jobs text entry
        ttk.Label(args_window, text="n_cores: Enter +'ve integer or 'all available'").pack(pady=10)
        n_jobs_entry = ttk.Entry(args_window, width=30)
        n_jobs_entry.pack(pady=5)
        n_jobs_entry.insert(0, "all available")  # Default value

        # Execute Button
        execute_button = ttk.Button(args_window, text="Run GRAPES Analysis",
                                    command=lambda: self.execute_grapes(
                                        normalised_by_var.get(),
                                        start_at_var.get(),
                                        pixel_size_entry.get(),
                                        fill_label_holes_var.get(),
                                        min_hole_size_entry.get(),
                                        anisotropy_entry.get(),
                                        parallel_entry.get(),
                                        n_jobs_entry.get(),
                                        args_window
                                    ))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(args_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(args_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def open_radial_quartiles_window(self, title):
        """Open a window to perform radial quartiles with specified parameters."""
        # Open a new window to enter arguments and display docstring
        args_window = tk.Toplevel(self)
        args_window.title(title)
        args_window.geometry("500x600")

        # Fetch the docstring
        docstring = grapes_module.radial_layers_quartiles.__doc__ or "No documentation available."

        # Parameter Inputs

        # 1.  Dropdown
        ttk.Label(args_window, text="Property:").pack(pady=10)
        prop_var = tk.StringVar()
        prop_combo = ttk.Combobox(args_window, textvariable=prop_var)
        prop_combo['values'] = ("volume",
                                "centroid-0",
                                "centroid-1",
                                "centroid-2",
                                "equivalent_diameter_volume",
                                "intensity_max",
                                "intensity_mean",
                                "intensity_min",
                                "_std",
                                "_surface_area",
                                "sphericity")
        prop_combo.current(0)
        prop_combo.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(args_window, text="Run Radial Quartiles Analysis",
                                    command=lambda: self.execute_radial_quartiles(prop_var.get(), args_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(args_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(args_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def open_radial_deciles_window(self, title):
        """Open a window to perform radial deciles with specified parameters."""
        # Open a new window to enter arguments and display docstring
        args_window = tk.Toplevel(self)
        args_window.title(title)
        args_window.geometry("500x600")

        # Fetch the docstring
        docstring = grapes_module.radial_layers_deciles.__doc__ or "No documentation available."

        # Parameter Inputs

        # 1.  Dropdown
        ttk.Label(args_window, text="Property:").pack(pady=10)
        prop_var = tk.StringVar()
        prop_combo = ttk.Combobox(args_window, textvariable=prop_var)
        prop_combo['values'] = ("volume",
                                "centroid-0",
                                "centroid-1",
                                "centroid-2",
                                "equivalent_diameter_volume",
                                "intensity_max",
                                "intensity_mean",
                                "intensity_min",
                                "_std",
                                "_surface_area",
                                "sphericity")
        prop_combo.current(0)
        prop_combo.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(args_window, text="Run Radial Deciles Analysis",
                                    command=lambda: self.execute_radial_deciles(prop_var.get(), args_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(args_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(args_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def open_quartiles_window(self, title):
        """Open a window to perform quartiles with specified parameters."""
        # Open a new window to enter arguments and display docstring
        args_window = tk.Toplevel(self)
        args_window.title(title)
        args_window.geometry("500x600")

        # Fetch the docstring
        docstring = grapes_module.compute_quartile_statistics.__doc__ or "No documentation available."

        # Parameter Inputs

        # 1.  Dropdown
        ttk.Label(args_window, text="Property:").pack(pady=10)
        prop_var = tk.StringVar()
        prop_combo = ttk.Combobox(args_window, textvariable=prop_var)
        prop_combo['values'] = ("volume",
                                "centroid-0",
                                "centroid-1",
                                "centroid-2",
                                "equivalent_diameter_volume",
                                "intensity_max",
                                "intensity_mean",
                                "intensity_min",
                                "_std",
                                "_surface_area",
                                "sphericity")
        prop_combo.current(0)
        prop_combo.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(args_window, text="Run Quartiles Analysis",
                                    command=lambda: self.execute_quartiles(prop_var.get(), args_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(args_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(args_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def open_deciles_window(self, title):
        """Open a window to perform deciles with specified parameters."""
        # Open a new window to enter arguments and display docstring
        args_window = tk.Toplevel(self)
        args_window.title(title)
        args_window.geometry("500x600")

        # Fetch the docstring
        docstring = grapes_module.compute_decile_statistics.__doc__ or "No documentation available."

        # Parameter Inputs

        # 1.  Dropdown
        ttk.Label(args_window, text="Property:").pack(pady=10)
        prop_var = tk.StringVar()
        prop_combo = ttk.Combobox(args_window, textvariable=prop_var)
        prop_combo['values'] = ("volume",
                                "centroid-0",
                                "centroid-1",
                                "centroid-2",
                                "equivalent_diameter_volume",
                                "intensity_max",
                                "intensity_mean",
                                "intensity_min",
                                "_std",
                                "_surface_area",
                                "sphericity")
        prop_combo.current(0)
        prop_combo.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(args_window, text="Run Deciles Analysis",
                                    command=lambda: self.execute_deciles(prop_var.get(), args_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(args_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(args_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def open_prop_2_image_window(self, title):
        """Open a window to perform prop_2_image with specified parameters."""
        # Open a new window to enter arguments and display docstring
        args_window = tk.Toplevel(self)
        args_window.title(title)
        args_window.geometry("500x600")

        # Fetch the docstring
        docstring = grapes_module.prop_2_image.__doc__ or "No documentation available."

        # Parameter Inputs

        # 1.  Dropdown
        ttk.Label(args_window, text="Property:").pack(pady=10)
        prop_var = tk.StringVar()
        prop_combo = ttk.Combobox(args_window, textvariable=prop_var)
        prop_combo['values'] = ("volume",
                                "centroid-0",
                                "centroid-1",
                                "centroid-2",
                                "equivalent_diameter_volume",
                                "intensity_max",
                                "intensity_mean",
                                "intensity_min",
                                "_std",
                                "_surface_area",
                                "sphericity")
        prop_combo.current(0)
        prop_combo.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(args_window, text="Run Assign Property to Labels Analysis",
                                    command=lambda: self.execute_prop_2_image(prop_var.get(), args_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(args_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(args_window, wrap=tk.WORD, height=10, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def execute_grapes(
        self, 
        normalised_by, 
        start_at, 
        pixel_size, 
        fill_label_holes, 
        min_hole_size, 
        anisotropy,
        parallel,
        n_jobs,
        window
        ):
        """Execute the GRAPES analysis with provided parameters."""
        # Validate and convert inputs
        try:
            # Convert pixel_size to float if provided
            if pixel_size.strip() == "":
                pixel_size_val = None
            else:
                pixel_size_val = float(pixel_size)

            # Convert min_hole_size to int
            min_hole_size_val = int(min_hole_size)

            # Validate normalised_by and start_at
            if normalised_by not in ["radial_max", "surface"]:
                raise ValueError("Invalid value for 'normalised_by'. Choose 'radial_max' or 'surface'.")

            if start_at not in ["edge", "center"]:
                raise ValueError("Invalid value for 'start_at'. Choose 'edge' or 'center'.")

            # Validate anisotropy
            anisotropy = ast.literal_eval(anisotropy)
            # Check if anisotropy is a tuple of 2 or 3 values
            if not isinstance(anisotropy, tuple) or len(anisotropy) not in [2, 3]:
                raise ValueError("Invalid value for 'anisotropy'. Enter a tuple of 2 or 3 values.")

            # Validate parallel
            parallel = int(parallel)
            if parallel <= 0:
                raise ValueError("Invalid value for 'parallel'. Enter a positive integer.")
            
            # Validate n_jobs
            if n_jobs == "all available":
                n_jobs = -1
            else:
                n_jobs = int(n_jobs)

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            return

        # Execute the GRAPES analysis
        result = self.grapes.calc_GRAPES(
            normalised_by=normalised_by,
            start_at=start_at,
            pixel_size=pixel_size_val,
            fill_label_holes=fill_label_holes,
            min_hole_size=min_hole_size_val,
            anisotropy=anisotropy,
            parallel=parallel,
            n_jobs=n_jobs
        )

        # Display the result
        if isinstance(result, str):
            # If result is a string (e.g., error message or confirmation)
            messagebox.showinfo("GRAPES Analysis", result)
        elif isinstance(result, pd.DataFrame):
            # If result is a DataFrame, you might want to display it or save it
            self.grapes.grapes_df = result
            messagebox.showinfo("GRAPES Analysis", f"GRAPES Analysis Completed. DataFrame with {len(result)} records created.")
        else:
            messagebox.showinfo("GRAPES Analysis", "GRAPES Analysis Completed.")

        # Close the window
        window.destroy()

    def execute_radial_quartiles(self, prop, window):
        """Execute the GRAPES analysis with provided parameters."""
        # Validate and convert inputs
        try:
            # Validate normalised_by and start_at
            if prop not in ["volume",
                            "centroid-0",
                            "centroid-1",
                            "centroid-2",
                            "equivalent_diameter_volume",
                            "intensity_max",
                            "intensity_mean",
                            "intensity_min",
                            "_std",
                            "_surface_area",
                            "sphericity"]:
                raise ValueError("Invalid value for 'prop'. Choose a valid property.")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            return

        # Execute the GRAPES analysis
        result = self.grapes.calc_radial_layers_quartiles(prop=prop)

        # Display the result
        if isinstance(result, str):
            # If result is a string (e.g., error message or confirmation)
            messagebox.showinfo("Radial Quartiles Analysis", result)
        elif isinstance(result, pd.DataFrame):
            # If result is a DataFrame, later we want to save it
            self.grapes.radial_quartiles_df = result
            messagebox.showinfo("Radial Quartiles Analysis", f"Radial Quartiles Analysis Completed. DataFrame with {len(result)} records created.")
        else:
            messagebox.showinfo("Radial Quartiles Analysis", "Radial Quartiles Analysis Completed.")

        # Close the window
        window.destroy()
    
    def execute_radial_deciles(self, prop, window):
        """Execute the GRAPES analysis with provided parameters."""
        # Validate and convert inputs
        try:
            # Validate normalised_by and start_at
            if prop not in ["volume",
                            "centroid-0",
                            "centroid-1",
                            "centroid-2",
                            "equivalent_diameter_volume",
                            "intensity_max",
                            "intensity_mean",
                            "intensity_min",
                            "_std",
                            "_surface_area",
                            "sphericity"]:
                raise ValueError("Invalid value for 'prop'. Choose a valid property.")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            return

        # Execute the GRAPES analysis
        result = self.grapes.calc_radial_layers_deciles(prop=prop)

        # Display the result
        if isinstance(result, str):
            # If result is a string (e.g., error message or confirmation)
            messagebox.showinfo("Radial Deciles Analysis", result)
        elif isinstance(result, pd.DataFrame):
            # If result is a DataFrame, later we want to save it
            self.grapes.radial_deciles_df = result
            messagebox.showinfo("Radial Deciles Analysis", f"Radial Deciles Analysis Completed. DataFrame with {len(result)} records created.")
        else:
            messagebox.showinfo("Radial Deciles Analysis", "Radial Deciles Analysis Completed.")

        # Close the window
        window.destroy()

    def execute_quartiles(self, prop, window):
        """Execute the GRAPES analysis with provided parameters."""
        # Validate and convert inputs
        try:
            # Validate normalised_by and start_at
            if prop not in ["volume",
                            "centroid-0",
                            "centroid-1",
                            "centroid-2",
                            "equivalent_diameter_volume",
                            "intensity_max",
                            "intensity_mean",
                            "intensity_min",
                            "_std",
                            "_surface_area",
                            "sphericity"]:
                raise ValueError("Invalid value for 'prop'. Choose a valid property.")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            return

        # Execute the GRAPES analysis
        result = self.grapes.calc_quartiles(prop=prop)

        # Display the result
        if isinstance(result, str):
            # If result is a string (e.g., error message or confirmation)
            messagebox.showinfo("Quartiles Analysis", result)
        elif isinstance(result, pd.DataFrame):
            # If result is a DataFrame, later we want to save it
            self.grapes.quartiles_df = result
            messagebox.showinfo("Quartiles Analysis", f"Quartiles Analysis Completed. DataFrame with {len(result)} records created.")
        else:
            messagebox.showinfo("Quartiles Analysis", "Quartiles Analysis Completed.")

        # Close the window
        window.destroy()

    def execute_deciles(self, prop, window):
        """Execute the GRAPES analysis with provided parameters."""
        # Validate and convert inputs
        try:
            # Validate normalised_by and start_at
            if prop not in ["volume",
                            "centroid-0",
                            "centroid-1",
                            "centroid-2",
                            "equivalent_diameter_volume",
                            "intensity_max",
                            "intensity_mean",
                            "intensity_min",
                            "_std",
                            "_surface_area",
                            "sphericity"]:
                raise ValueError("Invalid value for 'prop'. Choose a valid property.")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            return

        # Execute the GRAPES analysis
        result = self.grapes.calc_deciles(prop=prop)

        # Display the result
        if isinstance(result, str):
            # If result is a string (e.g., error message or confirmation)
            messagebox.showinfo("Deciles Analysis", result)
        elif isinstance(result, pd.DataFrame):
            # If result is a DataFrame, later we want to save it
            self.grapes.deciles_df = result
            messagebox.showinfo("Deciles Analysis", f"Deciles Analysis Completed. DataFrame with {len(result)} records created.")
        else:
            messagebox.showinfo("Deciles Analysis", "Deciles Analysis Completed.")

        # Close the window
        window.destroy()

    def execute_prop_2_image(self, prop, window):
        """Execute the GRAPES analysis with provided parameters."""
        # Validate and convert inputs
        try:
            # Validate normalised_by and start_at
            if prop not in ["volume",
                            "centroid-0",
                            "centroid-1",
                            "centroid-2",
                            "equivalent_diameter_volume",
                            "intensity_max",
                            "intensity_mean",
                            "intensity_min",
                            "_std",
                            "_surface_area",
                            "sphericity"]:
                raise ValueError("Invalid value for 'prop'. Choose a valid property.")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            return

        # Execute the GRAPES analysis
        result = self.grapes.prop_2_image(prop=prop)

        # Display the result
        if isinstance(result, str):
            # If result is a string (e.g., error message or confirmation)
            messagebox.showinfo("Assign Property to Labels Analysis", result)
        elif isinstance(result, np.ndarray):
            # If result is a ndarray, later we want to save it
            self.grapes.prop_image = result
            messagebox.showinfo("Assign Property to Labels Analysis", f"Assign Property to Labels Analysis Completed. Image with shape {result.shape} created.")
        else:
            messagebox.showinfo("Assign Property to Labels Analysis", "Warning: Assign Property to Labels Analysis Completed without Creating Image.")

        # Close the window
        window.destroy()

##########################################################################################
# Plotting Tab
##########################################################################################

class PlottingTab(ttk.Frame):
    def __init__(self, parent, grapes):
        super().__init__(parent)
        self.grapes = grapes

        # Existing plotting functions
        self.plot_functions = {
            "Plot Particle Images": self.plot_particle_image,
            "Plot Quartile Radial Layers": self.plot_quartile_radial_layers,
            "Plot Decile Radial Layers": self.plot_decile_radial_layers,
            "Plot Particle Radial Intensity": self.plot_radial_intensities,
            "Plot Volume Slices": self.plot_vol_slice_images
            # Add more functions here as needed
        }

        # Create Grid of Buttons (3x2 if adding more functions)
        self.create_buttons_grid()

    def create_buttons_grid(self):
        functions = list(self.plot_functions.keys())
        total_functions = len(functions)
        cols = 2
        rows = (total_functions + cols - 1) // cols  # Ceiling division

        for index, func_name in enumerate(functions):
            row = index // cols
            col = index % cols
            button = ttk.Button(self, text=func_name, command=self.plot_functions[func_name])
            button.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')

        # Configure grid weights for responsiveness
        for i in range(rows):
            self.grid_rowconfigure(i, weight=1)
        for j in range(cols):
            self.grid_columnconfigure(j, weight=1)

    # Existing plot methods
    def plot_particle_image(self):
        self.open_particle_image_window("Plot Particle Image")
    
    def plot_quartile_radial_layers(self):
        self.open_quartile_radial_layers_window("Plot Quartile Radial Layers")

    def plot_decile_radial_layers(self):
        self.open_decile_radial_layers_window("Plot Decile Radial Layers")

    def plot_radial_intensities(self):
        self.open_radial_intensities_window("Plot Particle Radial Intensity")

    def plot_vol_slice_images(self):
        self.open_vol_slice_images_window("Plot Volume Slices")
        # Add more plot methods here as needed

    def open_particle_image_window(self, title):
        """Open a window to input parameters for plotting particle images."""
        plot_window = tk.Toplevel(self)
        plot_window.title(title)
        plot_window.geometry("500x700")

        # Fetch the docstring of plot_particle_image
        docstring = self.grapes.plot_particle_image_method.__doc__ or "No documentation available."

        # Label Entry
        ttk.Label(plot_window, text="Label:").pack(pady=10)
        self.label_entry = ttk.Entry(plot_window, width=20)
        self.label_entry.pack(pady=5)

        # Image Type Selection
        ttk.Label(plot_window, text="Image Type:").pack(pady=10)
        self.image_type_var = tk.StringVar(value='distance_transform')
        image_types = ['distance_transform', 'graylevel', 'mask']
        self.image_type_menu = ttk.OptionMenu(plot_window, self.image_type_var, 'distance_transform', *image_types)
        self.image_type_menu.pack(pady=5)

        # Slice Index Entry
        ttk.Label(plot_window, text="Slice Index (Optional):").pack(pady=10)
        self.slice_idx_entry = ttk.Entry(plot_window, width=20)
        self.slice_idx_entry.pack(pady=5)

        # Title Suffix Entry
        ttk.Label(plot_window, text="Title Suffix (Optional):").pack(pady=10)
        self.title_suffix_entry = ttk.Entry(plot_window, width=20)
        self.title_suffix_entry.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(plot_window, text="Generate Plot",
                                    command=lambda: self.execute_particle_plot(plot_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(plot_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(plot_window, wrap=tk.WORD, height=15, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def execute_particle_plot(self, window):
        """Execute the plot_particle_image function with provided parameters."""
        # Retrieve user inputs
        label_str = self.label_entry.get().strip()
        image_type = self.image_type_var.get()
        slice_idx_str = self.slice_idx_entry.get().strip()
        title_suffix = self.title_suffix_entry.get().strip()

        # Input validation
        if not label_str.isdigit():
            messagebox.showerror("Input Error", "Label must be an integer.")
            return
        label = int(label_str)

        slice_idx: Optional[int] = None
        if slice_idx_str:
            if not slice_idx_str.isdigit():
                messagebox.showerror("Input Error", "Slice Index must be an integer.")
                return
            slice_idx = int(slice_idx_str)

        if not title_suffix:
            title_suffix = None  # Handle empty string as None

        # Generate the plot using GRAPES class
        try:
            fig: Optional[Figure] = self.grapes.plot_particle_image_method(
                label=label,
                image_type=image_type,
                slice_idx=slice_idx,
                title_suffix=title_suffix,
                show=True 
            )
            if fig is None:
                messagebox.showerror("Plot Error", "Failed to generate the plot.")
                return
        except Exception as e:
            messagebox.showerror("Plot Error", f"An error occurred while plotting: {e}")
            return

        # Close the parameter window
        window.destroy()
    
    def open_quartile_radial_layers_window(self, title):
        """Open a window to input parameters for plotting quartile radial layers."""
        plot_window = tk.Toplevel(self)
        plot_window.title(title)
        plot_window.geometry("500x700")

        # Fetch the docstring of plot_quartile_radial_layers_method
        docstring = self.grapes.plot_quartile_radial_layers_method.__doc__ or "No documentation available."

        # Property Selection
        ttk.Label(plot_window, text="Property:").pack(pady=10)
        self.prop_var = tk.StringVar(value='radial_layers')  # Changed from 'property_var' to 'prop_var'
        properties = ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed']
        self.prop_menu = ttk.OptionMenu(plot_window, self.prop_var, 'radial_layers', *properties)
        self.prop_menu.pack(pady=5)

        # Error Type Selection
        ttk.Label(plot_window, text="Error Type:").pack(pady=10)
        self.error_type_var = tk.StringVar(value='SE')
        error_types = ['SE', 'Std']
        self.error_type_menu = ttk.OptionMenu(plot_window, self.error_type_var, 'SE', *error_types)
        self.error_type_menu.pack(pady=5)

        # Title Entry
        ttk.Label(plot_window, text="Title (Optional):").pack(pady=10)
        self.title_entry = ttk.Entry(plot_window, width=30)
        self.title_entry.pack(pady=5)

        # X-axis Label Entry
        ttk.Label(plot_window, text="X-axis Label:").pack(pady=10)
        self.xlabel_entry = ttk.Entry(plot_window, width=30)
        self.xlabel_entry.insert(0, 'Radial Layer')  # Default value
        self.xlabel_entry.pack(pady=5)

        # Y-axis Label Entry
        ttk.Label(plot_window, text="Y-axis Label:").pack(pady=10)
        self.ylabel_entry = ttk.Entry(plot_window, width=30)
        self.ylabel_entry.insert(0, 'Mean Value')  # Default value
        self.ylabel_entry.pack(pady=5)

        # Figure Size Entry
        ttk.Label(plot_window, text="Figure Size (Width, Height):").pack(pady=10)
        self.figsize_entry = ttk.Entry(plot_window, width=30)
        self.figsize_entry.insert(0, '10,6')  # Default value
        self.figsize_entry.pack(pady=5)

        # Palette Entry
        ttk.Label(plot_window, text="Palette Colors (comma-separated, optional):").pack(pady=10)
        self.palette_entry = ttk.Entry(plot_window, width=30)
        self.palette_entry.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(plot_window, text="Generate Plot",
                                    command=lambda: self.execute_quartile_radial_layers_plot(plot_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(plot_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(plot_window, wrap=tk.WORD, height=15, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def execute_quartile_radial_layers_plot(self, window):
        """Execute the plot_quartile_radial_layers_method with provided parameters in a separate thread."""
        # Retrieve user inputs
        prop = self.prop_var.get()
        error_type = self.error_type_var.get()
        title = self.title_entry.get().strip()
        xlabel = self.xlabel_entry.get().strip()
        ylabel = self.ylabel_entry.get().strip()
        figsize_str = self.figsize_entry.get().strip()
        palette_str = self.palette_entry.get().strip()

        # Input validation and conversion
        # Convert figsize from string to tuple
        try:
            figsize = tuple(map(float, figsize_str.split(',')))
            if len(figsize) != 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Figure size must be two numbers separated by a comma (e.g., 10,6).")
            return

        # Convert palette from string to list
        if palette_str:
            palette = [color.strip() for color in palette_str.split(',')]
        else:
            palette = None

        # Define a function to generate the plot
        def generate_plot():
            try:
                fig = self.grapes.plot_quartile_radial_layers_method(
                    prop=prop,
                    error_type=error_type,
                    title=title if title else None,
                    xlabel=xlabel if xlabel else 'Radial Layer',
                    ylabel=ylabel if ylabel else 'Mean Value',
                    figsize=figsize,
                    palette=palette
                    # save_path parameter removed as per request
                )
                if fig:
                    messagebox.showinfo("Plot Generated", "Quartile Radial Layers plot has been generated successfully.")
                else:
                    messagebox.showerror("Plot Error", "Failed to generate the plot.")
            except Exception as e:
                messagebox.showerror("Plot Error", f"An error occurred while generating the plot: {e}")

        # Start the plot generation in a new thread to prevent GUI freezing
        threading.Thread(target=generate_plot).start()

        # Close the parameter window
        window.destroy()
    
    def open_decile_radial_layers_window(self, title):
        """Open a window to input parameters for plotting decile radial layers."""
        plot_window = tk.Toplevel(self)
        plot_window.title(title)
        plot_window.geometry("500x700")

        # Fetch the docstring of plot_decile_radial_layers_method
        docstring = self.grapes.plot_decile_radial_layers_method.__doc__ or "No documentation available."

        # Property Selection
        ttk.Label(plot_window, text="Property:").pack(pady=10)
        self.prop_var = tk.StringVar(value='radial_layers')  # Changed from 'property_var' to 'prop_var'
        properties = ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed']
        self.prop_menu = ttk.OptionMenu(plot_window, self.prop_var, 'radial_layers', *properties)
        self.prop_menu.pack(pady=5)

        # Error Type Selection
        ttk.Label(plot_window, text="Error Type:").pack(pady=10)
        self.error_type_var = tk.StringVar(value='SE')
        error_types = ['SE', 'Std']
        self.error_type_menu = ttk.OptionMenu(plot_window, self.error_type_var, 'SE', *error_types)
        self.error_type_menu.pack(pady=5)

        # Title Entry
        ttk.Label(plot_window, text="Title (Optional):").pack(pady=10)
        self.title_entry = ttk.Entry(plot_window, width=30)
        self.title_entry.pack(pady=5)

        # X-axis Label Entry
        ttk.Label(plot_window, text="X-axis Label:").pack(pady=10)
        self.xlabel_entry = ttk.Entry(plot_window, width=30)
        self.xlabel_entry.insert(0, 'Radial Layer')  # Default value
        self.xlabel_entry.pack(pady=5)

        # Y-axis Label Entry
        ttk.Label(plot_window, text="Y-axis Label:").pack(pady=10)
        self.ylabel_entry = ttk.Entry(plot_window, width=30)
        self.ylabel_entry.insert(0, 'Mean Value')  # Default value
        self.ylabel_entry.pack(pady=5)

        # Figure Size Entry
        ttk.Label(plot_window, text="Figure Size (Width, Height):").pack(pady=10)
        self.figsize_entry = ttk.Entry(plot_window, width=30)
        self.figsize_entry.insert(0, '10,6')  # Default value
        self.figsize_entry.pack(pady=5)

        # Palette Entry
        ttk.Label(plot_window, text="Palette Colors (comma-separated, optional):").pack(pady=10)
        self.palette_entry = ttk.Entry(plot_window, width=30)
        self.palette_entry.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(plot_window, text="Generate Plot",
                                    command=lambda: self.execute_decile_radial_layers_plot(plot_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(plot_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(plot_window, wrap=tk.WORD, height=15, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def execute_decile_radial_layers_plot(self, window):
        """Execute the plot_decile_radial_layers_method with provided parameters in a separate thread."""
        # Retrieve user inputs
        prop = self.prop_var.get()
        error_type = self.error_type_var.get()
        title = self.title_entry.get().strip()
        xlabel = self.xlabel_entry.get().strip()
        ylabel = self.ylabel_entry.get().strip()
        figsize_str = self.figsize_entry.get().strip()
        palette_str = self.palette_entry.get().strip()

        # Input validation and conversion
        # Convert figsize from string to tuple
        try:
            figsize = tuple(map(float, figsize_str.split(',')))
            if len(figsize) != 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Figure size must be two numbers separated by a comma (e.g., 10,6).")
            return

        # Convert palette from string to list
        if palette_str:
            palette = [color.strip() for color in palette_str.split(',')]
        else:
            palette = None

        # Define a function to generate the plot
        def generate_plot():
            try:
                fig = self.grapes.plot_decile_radial_layers_method(
                    prop=prop,
                    error_type=error_type,
                    title=title if title else None,
                    xlabel=xlabel if xlabel else 'Radial Layer',
                    ylabel=ylabel if ylabel else 'Mean Value',
                    figsize=figsize,
                    palette=palette

                )
                if fig:
                    messagebox.showinfo("Plot Generated", "Decile Radial Layers plot has been generated successfully.")
                else:
                    messagebox.showerror("Plot Error", "Failed to generate the plot.")
            except Exception as e:
                messagebox.showerror("Plot Error", f"An error occurred while generating the plot: {e}")

        # Start the plot generation in a new thread to prevent GUI freezing
        threading.Thread(target=generate_plot).start()

        # Close the parameter window
        window.destroy()

    def open_radial_intensities_window(self, title):
        """Open a window to input parameters for plotting radial intensities."""
        plot_window = tk.Toplevel(self)
        plot_window.title(title)
        plot_window.geometry("500x1000")

        # Fetch the docstring of plot_radial_intensities_method
        docstring = self.grapes.plot_radial_intensities_method.__doc__ or "No documentation available."

        # Label Entry
        ttk.Label(plot_window, text="Particle Label:").pack(pady=10)
        self.label_entry = ttk.Entry(plot_window, width=30)
        self.label_entry.pack(pady=5)

        # Plotting Type Selection
        ttk.Label(plot_window, text="Plotting Type:").pack(pady=10)
        self.plotting_var = tk.StringVar(value='normalised')
        plotting_types = ['normalised', 'graylevels']
        self.plotting_menu = ttk.OptionMenu(plot_window, self.plotting_var, 'normalised', *plotting_types)
        self.plotting_menu.pack(pady=5)

        # Title Suffix Entry
        ttk.Label(plot_window, text="Title Suffix (Optional):").pack(pady=10)
        self.title_suffix_entry = ttk.Entry(plot_window, width=30)
        self.title_suffix_entry.pack(pady=5)

        # X-axis Label Entry
        ttk.Label(plot_window, text="X-axis Label:").pack(pady=10)
        self.xlabel_entry = ttk.Entry(plot_window, width=30)
        self.xlabel_entry.insert(0, 'Radial Position')  # Default value
        self.xlabel_entry.pack(pady=5)

        # Y-axis Label Entry
        ttk.Label(plot_window, text="Y-axis Label (Optional):").pack(pady=10)
        self.ylabel_entry = ttk.Entry(plot_window, width=30)
        self.ylabel_entry.pack(pady=5)

        # Figure Size Entry
        ttk.Label(plot_window, text="Figure Size (Width, Height):").pack(pady=10)
        self.figsize_entry = ttk.Entry(plot_window, width=30)
        self.figsize_entry.insert(0, '8,6')  # Default value
        self.figsize_entry.pack(pady=5)

        # Color Entry
        ttk.Label(plot_window, text="Plot Color:").pack(pady=10)
        self.color_entry = ttk.Entry(plot_window, width=30)
        self.color_entry.insert(0, 'blue')  # Default value
        self.color_entry.pack(pady=5)

        # Marker Style Entry
        ttk.Label(plot_window, text="Marker Style (Optional):").pack(pady=10)
        self.marker_entry = ttk.Entry(plot_window, width=30)
        self.marker_entry.insert(0, 'o')  # Default value
        self.marker_entry.pack(pady=5)

        # Line Style Entry
        ttk.Label(plot_window, text="Line Style:").pack(pady=10)
        self.linestyle_entry = ttk.Entry(plot_window, width=30)
        self.linestyle_entry.insert(0, '-')  # Default value
        self.linestyle_entry.pack(pady=5)

        # Line Width Entry
        ttk.Label(plot_window, text="Line Width:").pack(pady=10)
        self.linewidth_entry = ttk.Entry(plot_window, width=30)
        self.linewidth_entry.insert(0, '1.5')  # Default value
        self.linewidth_entry.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(plot_window, text="Generate Plot",
                                    command=lambda: self.execute_radial_intensities_plot(plot_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(plot_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(plot_window, wrap=tk.WORD, height=15, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def execute_radial_intensities_plot(self, window):
        """Execute the plot_radial_intensities_method with provided parameters in a separate thread."""
        # Retrieve user inputs
        label_str = self.label_entry.get().strip()
        plotting = self.plotting_var.get()
        title_suffix = self.title_suffix_entry.get().strip()
        xlabel = self.xlabel_entry.get().strip()
        ylabel = self.ylabel_entry.get().strip()
        figsize_str = self.figsize_entry.get().strip()
        color = self.color_entry.get().strip()
        marker = self.marker_entry.get().strip()
        linestyle = self.linestyle_entry.get().strip()
        linewidth_str = self.linewidth_entry.get().strip()

        # Input validation and conversion
        # Convert label to integer
        try:
            label = int(label_str)
        except ValueError:
            messagebox.showerror("Input Error", "Particle Label must be an integer.")
            return

        # Validate plotting type
        if plotting not in ['normalised', 'graylevels']:
            messagebox.showerror("Input Error", "Plotting Type must be either 'normalised' or 'graylevels'.")
            return

        # Convert figsize from string to tuple
        try:
            figsize = tuple(map(float, figsize_str.split(',')))
            if len(figsize) != 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Figure size must be two numbers separated by a comma (e.g., 8,6).")
            return

        # Convert linewidth from string to float
        try:
            linewidth = float(linewidth_str)
        except ValueError:
            messagebox.showerror("Input Error", "Line Width must be a numeric value (e.g., 1.5).")
            return

        # Handle marker entry
        marker = marker if marker else None

        # Set default ylabel if not provided
        if not ylabel:
            ylabel = 'Normalised Graylevel' if plotting == 'normalised' else 'Graylevel'

        # Define a function to generate the plot
        def generate_plot():
            try:
                fig = self.grapes.plot_radial_intensities_method(
                    label=label,
                    plotting=plotting,
                    title_suffix=title_suffix if title_suffix else None,
                    xlabel=xlabel if xlabel else 'Radial Position',
                    ylabel=ylabel,
                    figsize=figsize,
                    color=color if color else 'blue',
                    marker=marker,
                    linestyle=linestyle if linestyle else '-',
                    linewidth=linewidth
                )
                if fig:
                    messagebox.showinfo("Plot Generated", "Radial Intensities plot has been generated successfully.")
                else:
                    messagebox.showerror("Plot Error", "Failed to generate the plot.")
            except Exception as e:
                messagebox.showerror("Plot Error", f"An error occurred while generating the plot: {e}")

        # Start the plot generation in a new thread to prevent GUI freezing
        threading.Thread(target=generate_plot).start()

        # Close the parameter window
        window.destroy()

    def open_vol_slice_images_window(self, title):
        """Open a window to input parameters for plotting particle images."""
        plot_window = tk.Toplevel(self)
        plot_window.title(title)
        plot_window.geometry("500x700")

        # Fetch the docstring of plot_particle_image
        docstring = grapes_module.plot_vol_slice_images.__doc__ or "No documentation available."

        # Image Type Selection
        ttk.Label(plot_window, text="Image Type:").pack(pady=10)
        self.image_type_var = tk.StringVar(value='Graylevel')
        image_types = ['Graylevel', 'Labels', 'Property2Image']
        self.image_type_menu = ttk.OptionMenu(plot_window, self.image_type_var, 'Graylevel', *image_types)
        self.image_type_menu.pack(pady=5)

        # Slice Index Entry
        ttk.Label(plot_window, text="Slice Index (Optional):").pack(pady=10)
        self.slice_idx_entry = ttk.Entry(plot_window, width=20)
        self.slice_idx_entry.pack(pady=5)

        # Title Suffix Entry
        ttk.Label(plot_window, text="Title Suffix (Optional):").pack(pady=10)
        self.title_suffix_entry = ttk.Entry(plot_window, width=20)
        self.title_suffix_entry.pack(pady=5)

        # Execute Button
        execute_button = ttk.Button(plot_window, text="Generate Plot",
                                    command=lambda: self.execute_volume_slice_plot(plot_window))
        execute_button.pack(pady=20)

        # Docstring Display
        ttk.Label(plot_window, text="Function Description:").pack(pady=10)
        doc_text = scrolledtext.ScrolledText(plot_window, wrap=tk.WORD, height=15, width=60, state='disabled')
        doc_text.pack(pady=5, padx=10, fill='both', expand=True)
        doc_text.config(state='normal')
        doc_text.insert(tk.END, docstring)
        doc_text.config(state='disabled')

    def execute_volume_slice_plot(self, window):
        """Execute the plot_volume_sice_image function with provided parameters."""
        # Retrieve user inputs
        image_type = self.image_type_var.get()
        slice_idx_str = self.slice_idx_entry.get().strip()
        title_suffix = self.title_suffix_entry.get().strip()

        # Input selection
        if image_type == 'Graylevel':
            im = self.grapes.grey_arr
        elif image_type == 'Labels':
            im = self.grapes.labels_arr
        elif image_type == 'Property2Image':
            im = self.grapes.prop_image

        slice_idx: Optional[int] = None
        if slice_idx_str:
            if not slice_idx_str.isdigit():
                messagebox.showerror("Input Error", "Slice Index must be an integer.")
                return
            slice_idx = int(slice_idx_str)

        if not title_suffix:
            title_suffix = None  # Handle empty string as None

        # Generate the plot using GRAPES class
        try:
            fig: Optional[Figure] = self.grapes.plot_vol_slice_images_method(
                im=im,
                slice_idx=slice_idx,
                title_suffix=title_suffix,
                show=True 
            )
            if fig is None:
                messagebox.showerror("Plot Error", "Failed to generate the plot.")
                return
        except Exception as e:
            messagebox.showerror("Plot Error", f"An error occurred while plotting: {e}")
            return

        # Close the parameter window
        window.destroy()

if __name__ == "__main__":
    app = GrapesGUI()
    app.mainloop()
