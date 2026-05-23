# pubs-gc-ltreb

Public documentation, analysis, and code products for the global change LTREB projects at the Jornada Basin in southern New Mexico, U.S.A.

**Directories**

```
├── manuscripts                  - Analysis scripts, figure creation code, etc. for manuscripts
│   ├── schadenfreude            -   Materials for in-review "Schadenfreude" project MS
│   │   ├── SoilMoisture.ipynb   -     Jupyter notebook for a soil moisture analysis
│   │   └── ...
│   ├── data_tmp                 -   Temporary data file storage
│   ├── out_tmp                  -   Temporary data outputs
│   └── ...
├── metadata                     - Files with contextual information and metadata
│   ├── lt_plot_trt.csv          -   Plot and treatment table
│   └── ...
├── src                          - Common source code for loading or transforming project data
│   ├── schadenfreude_helper.py  -   Code for the Schadenfreude project
│   └── ...
├── environment.yml              - Conda environment YAML file
└── README.md                    - This file
```

**Branches**

The repository is organized into branches that have a series of named releases along them. These releases are what get published with Zenodo, usually with the intent of being cited in manuscripts. Current branches:

* `schadenf_submission` is for an in-review manuscript from the "Schadenfreude" project.

## System requirements

Most analyses are written in Python and are presented as Jupyter notebooks. Python packages required to run the analyses include:

    python >= 3.12.7
    pandas >= 2.3.1
    numpy >= 2.3.1
    seaborn >= 0.13.2
    matplotlib >=3.10.0

For full reproducibility, the Python software environment is managed by [Conda](https://conda.org), and the `environment.yml` file included here describes the software environment used to run all Python code and notebooks at the time of submission. In addition, some notebooks contain source code in R. Required R packages include:

    R >= 4.5.0
    tidyverse >= 2.0.0
    lavaan >= 0.6-21
    semPlot >= 1.1.8

## Installing and running the code

### Locally

Installing the requirements above should allow all notebooks to run locally or in a cloud environment like [Google Colab](https://colab.research.google.com). To run the notebooks locally, you will need to use Git to clone this repository to your local machine, then [install Conda](https://docs.conda.io/projects/conda/en/stable/index.html) and create the Conda environment defined in `environment.yml`, and then run the notebooks using JupyterLab (or a jupyter-friendly IDE like VSCode). The instructions below should work well in MacOS or Linux.

To clone the repository from your system shell issue:

    git clone https://github.com/jornada-im/pubs-gc-ltreb.git. # or use the SSH method if you prefer
    cd pubs-gc-ltreb                                           # enter the directory

Assuming you have Conda installed already, create the environment from `environment.yml` with:

    conda create --name myenv --file environment.yml

Activate the environment and run Jupyter Lab to access the notebooks.

    conda activate myenv 
    jupyter lab manuscripts/schadenfreude       # to access notebooks for the Schadenfreude MS

### In a cloud environment

To run in a cloud environment, open a browser window with the appropriate Google Colab path for this GitHub repository:

    https://colab.research.google.com/github/jornada-im/pubs-gc-ltreb

From here you can run the notebooks, but be aware that you'll need to add a new cell at the start of each notebook to clone the GitHub repository into the Colab `content` directory. This cell should contain:

    !git clone https://github.com/jornada-im/pubs-gc-ltreb.git

After the repository is cloned, edit any file paths used in the notebook for loading modules or data to point to `content/pubs-gc-ltreb/`. For example, the relative path used to load the file at `../../metadata/lt_plot_trt.csv` should be changed to `/content/pubs-gc-ltreb/metadata/lt_plot_trt.csv`. After this, the notebooks should run as normal as long as all project data are also in the `content` directory (see below).

## Data for the analyses

Most data used by the Jupyter notebooks and other code here have been published as datasets in the [EDI repository](https://portal.edirepository.org), and URLs or DOIs are provided. To work with these data as specified in the code, download required files from the specified location and save them to the `manuscripts/data_tmp` directory. If you are working in a cloud environment, each notebook's relative directory paths to the data will need to be altered (such as to `/content/jornada-im/pubs-gc-ltreb/manuscripts/data_tmp/` in the case of Colab).


## License

All content of the [jornada-im/pubs-gc-ltreb](https://github.com/jornada-im/pubs-gc-ltreb) repository &copy; Greg Maurer and Osvaldo Sala 2025-2026 and licensed under the [GNU Lesser General Public License 3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).