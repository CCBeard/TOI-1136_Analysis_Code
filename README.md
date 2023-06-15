# TOI-1136_Analysis_Code
Github repository to share analysis code used in Beard et al. 2023 for any who wish to reproduce, or modify, results.


My results were run in an anaconda environment on the UCI HPC cluster. The following software packages and their versions were installed:


# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             4.5                       1_gnu  
_r-mutex                  1.0.1               anacondar_1    conda-forge
aesara-theano-fallback    0.0.4                    pypi_0    pypi
alabaster                 0.7.12                   pypi_0    pypi
argparse                  1.4.0                    pypi_0    pypi
arviz                     0.11.2                   pypi_0    pypi
asteval                   0.9.28                   pypi_0    pypi
astropy                   5.1                      pypi_0    pypi
astroquery                0.4.3                    pypi_0    pypi
async-generator           1.10                     pypi_0    pypi
attrs                     21.2.0                   pypi_0    pypi
autograd                  1.3                      pypi_0    pypi
babel                     2.9.1                    pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
batman-package            2.4.8                    pypi_0    pypi
beautifulsoup4            4.9.3                    pypi_0    pypi
binutils_impl_linux-64    2.35.1               h27ae35d_9  
binutils_linux-64         2.35                h67ddf6f_30    conda-forge
bleach                    4.0.0                    pypi_0    pypi
bokeh                     2.3.3                    pypi_0    pypi
brokenaxes                0.4.2                    pypi_0    pypi
bwidget                   1.9.14               ha770c72_1    conda-forge
bzip2                     1.0.8                h7f98852_4    conda-forge
c-ares                    1.17.1               h7f98852_1    conda-forge
ca-certificates           2022.12.7            ha878542_0    conda-forge
cachetools                4.2.2                    pypi_0    pypi
cairo                     1.16.0               hf32fb01_1  
celerite                  0.4.2                    pypi_0    pypi
celerite2                 0.2.0                    pypi_0    pypi
certifi                   2021.5.30        py39h06a4308_0  
cffi                      1.14.6                   pypi_0    pypi
cftime                    1.5.0                    pypi_0    pypi
charset-normalizer        2.0.4                    pypi_0    pypi
cmake                     3.22.5                   pypi_0    pypi
configparser              5.0.2                    pypi_0    pypi
corner                    2.2.1                    pypi_0    pypi
cryptography              3.4.7                    pypi_0    pypi
curl                      7.76.1               h979ede3_1    conda-forge
cycler                    0.10.0                   pypi_0    pypi
cython                    0.29.24                  pypi_0    pypi
debugpy                   1.4.1                    pypi_0    pypi
decorator                 5.0.9                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
dill                      0.3.6                    pypi_0    pypi
docutils                  0.17.1                   pypi_0    pypi
dynesty                   1.2.3                    pypi_0    pypi
emcee                     3.1.0                    pypi_0    pypi
entrypoints               0.3                      pypi_0    pypi
exceptiongroup            1.0.1                    pypi_0    pypi
exoplanet                 0.5.1                    pypi_0    pypi
exoplanet-core            0.1.2                    pypi_0    pypi
fastprogress              1.0.0                    pypi_0    pypi
fbpca                     1.0                      pypi_0    pypi
filelock                  3.0.12                   pypi_0    pypi
fontconfig                2.13.1               h6c09931_0  
freetype                  2.11.0               h70c0345_0  
fribidi                   1.0.10               h7b6447c_0  
future                    0.18.2                   pypi_0    pypi
gcc_impl_linux-64         7.5.0               hda68d29_13    conda-forge
gcc_linux-64              7.5.0               h47867f9_30    conda-forge
george                    0.4.0                    pypi_0    pypi
gfortran_impl_linux-64    7.5.0               h56cb351_19    conda-forge
gfortran_linux-64         7.5.0               h78c8a43_30    conda-forge
glib                      2.69.1               h5202010_0  
graphite2                 1.3.14               h23475e2_0  
gsl                       2.4               h294904e_1006    conda-forge
gxx_impl_linux-64         7.5.0               h64c220c_13    conda-forge
gxx_linux-64              7.5.0               h555fc39_30    conda-forge
h5py                      3.3.0                    pypi_0    pypi
harfbuzz                  2.8.1                h6f93f22_0  
html5lib                  1.1                      pypi_0    pypi
icu                       58.2                 he6710b0_3  
idna                      3.2                      pypi_0    pypi
imagesize                 1.2.0                    pypi_0    pypi
importlib-metadata        4.6.3                    pypi_0    pypi
iniconfig                 1.1.1                    pypi_0    pypi
ipykernel                 6.0.3                    pypi_0    pypi
ipython                   7.26.0                   pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
jedi                      0.18.0                   pypi_0    pypi
jeepney                   0.7.1                    pypi_0    pypi
jinja2                    3.0.1                    pypi_0    pypi
joblib                    1.0.1                    pypi_0    pypi
jpeg                      9d                   h7f8727e_0  
jsonschema                3.2.0                    pypi_0    pypi
juliet                    2.1.2                    pypi_0    pypi
jupyter-client            6.1.12                   pypi_0    pypi
jupyter-core              4.7.1                    pypi_0    pypi
jupyterlab-pygments       0.1.2                    pypi_0    pypi
kernel-headers_linux-64   2.6.32              he073ed8_15    conda-forge
keyring                   23.0.1                   pypi_0    pypi
kiwisolver                1.3.1                    pypi_0    pypi
krb5                      1.17.2               h926e7f8_0    conda-forge
lcms2                     2.12                 hddcbb42_0    conda-forge
ld_impl_linux-64          2.35.1               h7274673_9  
libblas                   3.9.0                8_openblas    conda-forge
libcblas                  3.9.0                8_openblas    conda-forge
libcurl                   7.76.1               hc4aaa36_1    conda-forge
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libev                     4.33                 h516909a_1    conda-forge
libffi                    3.3                  he6710b0_2  
libgcc                    7.2.0                h69d50b8_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               h14aa051_19    conda-forge
libgfortran4              7.5.0               h14aa051_19    conda-forge
libgomp                   11.2.0               h1234567_1  
liblapack                 3.9.0                8_openblas    conda-forge
libllvm11                 11.1.0               hf817b99_2    conda-forge
libnghttp2                1.43.0               h812cca2_0    conda-forge
libopenblas               0.3.12          pthreads_hb3c22a3_1    conda-forge
libpng                    1.6.37               hbc83047_0  
libssh2                   1.9.0                ha56f1ee_6    conda-forge
libstdcxx-ng              12.2.0              h46fd767_19    conda-forge
libtiff                   4.2.0                h85742a9_0  
libuuid                   1.0.3                h7f8727e_2  
libwebp-base              1.2.0                h27cfd23_0  
libxcb                    1.14                 h7b6447c_0  
libxml2                   2.9.12               h03d6c58_0  
lightkurve                2.0.10                   pypi_0    pypi
llvmlite                  0.37.0                   pypi_0    pypi
lmfit                     1.0.3                    pypi_0    pypi
lz4-c                     1.9.3                h295c915_1  
make                      4.3                  hd18ef5c_1    conda-forge
markupsafe                2.0.1                    pypi_0    pypi
matplotlib                3.4.2                    pypi_0    pypi
matplotlib-base           3.3.4            py39h2fa2bec_0    conda-forge
matplotlib-inline         0.1.2                    pypi_0    pypi
memoization               0.4.0                    pypi_0    pypi
mimeparse                 0.1.3                    pypi_0    pypi
mistune                   0.8.4                    pypi_0    pypi
mpi                       1.0                       mpich    conda-forge
mpich                     3.2.1             hc99cbb1_1014    conda-forge
multinest                 3.10                 hab63836_5    conda-forge
multiprocess              0.70.14                  pypi_0    pypi
nbclient                  0.5.3                    pypi_0    pypi
nbconvert                 6.1.0                    pypi_0    pypi
nbformat                  5.1.3                    pypi_0    pypi
nbsphinx                  0.8.6                    pypi_0    pypi
ncurses                   6.2                  he6710b0_1  
nest-asyncio              1.5.1                    pypi_0    pypi
netcdf4                   1.5.7                    pypi_0    pypi
numba                     0.54.1                   pypi_0    pypi
numdifftools              0.9.41                   pypi_0    pypi
numpy                     1.20.3                   pypi_0    pypi
oktopus                   0.1.2                    pypi_0    pypi
olefile                   0.46               pyh9f0ad1d_1    conda-forge
openssl                   1.1.1k               h27cfd23_0  
packaging                 21.0                     pypi_0    pypi
pandas                    1.3.1                    pypi_0    pypi
pandocfilters             1.4.3                    pypi_0    pypi
pango                     1.45.3               hd140c19_0  
parso                     0.8.2                    pypi_0    pypi
patsy                     0.5.1                    pypi_0    pypi
pcre                      8.45                 h295c915_0  
pexpect                   4.8.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    8.3.1                    pypi_0    pypi
pip                       21.1.3           py39h06a4308_0  
pixman                    0.40.0               h7f8727e_1  
pluggy                    1.0.0                    pypi_0    pypi
prompt-toolkit            3.0.19                   pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pycparser                 2.20                     pypi_0    pypi
pyerfa                    2.0.0                    pypi_0    pypi
pygments                  2.9.0                    pypi_0    pypi
pymc3                     3.11.2                   pypi_0    pypi
pymc3-ext                 0.1.0                    pypi_0    pypi
pymultinest               2.11             py39hf3d152e_1    conda-forge
pyparsing                 2.4.7                    pypi_0    pypi
pyrsistent                0.18.0                   pypi_0    pypi
pytest                    7.2.0                    pypi_0    pypi
python                    3.9.6                h12debd9_0  
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python_abi                3.9                      2_cp39    conda-forge
pytz                      2021.1                   pypi_0    pypi
pyvo                      1.1                      pypi_0    pypi
pyyaml                    5.4.1                    pypi_0    pypi
pyzmq                     22.1.0                   pypi_0    pypi
r-base                    3.6.1                haffb61f_2  
r-bh                      1.69.0_1          r36h6115d3f_0    r
r-celestial               1.4.6             r36h6115d3f_2    conda-forge
r-crayon                  1.3.4             r36h6115d3f_0    r
r-digest                  0.6.18            r36h96ca727_0    r
r-dotcall64               1.0_0           r36h31ca83e_1006    conda-forge
r-fields                  11.6              r36h31ca83e_1    conda-forge
r-htmltools               0.3.6             r36h29659fb_0    r
r-httpuv                  1.5.1             r36h29659fb_0    r
r-jsonlite                1.6               r36h96ca727_0    r
r-later                   0.8.0             r36h29659fb_0    r
r-magicaxis               2.2.1             r36hc72bb7e_0    conda-forge
r-magrittr                1.5               r36h6115d3f_4    r
r-mapproj                 1.2.7             r36hcdcec82_1    conda-forge
r-maps                    3.3.0           r36hcdcec82_1004    conda-forge
r-mass                    7.3_54            r36hcfec24a_0    conda-forge
r-mime                    0.6               r36h96ca727_0    r
r-minpack.lm              1.2_1           r36hed91ed1_1007    conda-forge
r-nistunits               1.0.1           r36h6115d3f_1003    conda-forge
r-plotrix                 3.8_1             r36hc72bb7e_0    conda-forge
r-pracma                  2.3.3             r36hc72bb7e_0    conda-forge
r-promises                1.0.1             r36h29659fb_0    r
r-r6                      2.4.0             r36h6115d3f_0    r
r-rann                    2.6.1             r36h0357c0b_2    conda-forge
r-rcolorbrewer            1.1_2           r36h6115d3f_1003    conda-forge
r-rcpp                    1.0.1             r36h29659fb_0    r
r-rlang                   0.3.4             r36h96ca727_0    r
r-shiny                   1.3.2             r36h6115d3f_0    r
r-sm                      2.2_5.6         r36h31ca83e_1004    conda-forge
r-sourcetools             0.1.7             r36h29659fb_0    r
r-spam                    2.5_1             r36h9bbef5b_0    conda-forge
r-xtable                  1.8_4             r36h6115d3f_0    r
radvel                    1.4.7                    pypi_0    pypi
readline                  8.1                  h27cfd23_0  
requests                  2.26.0                   pypi_0    pypi
scikit-learn              0.24.2                   pypi_0    pypi
scipy                     1.9.2                    pypi_0    pypi
seaborn                   0.12.1                   pypi_0    pypi
secretstorage             3.3.1                    pypi_0    pypi
semver                    2.13.0                   pypi_0    pypi
setuptools                52.0.0           py39h06a4308_0  
six                       1.16.0             pyh6c4a22f_0    conda-forge
sklearn                   0.0.post1                pypi_0    pypi
snowballstemmer           2.1.0                    pypi_0    pypi
soupsieve                 2.2.1                    pypi_0    pypi
sphinx                    4.1.2                    pypi_0    pypi
sphinxcontrib-applehelp   1.0.2                    pypi_0    pypi
sphinxcontrib-devhelp     1.0.2                    pypi_0    pypi
sphinxcontrib-htmlhelp    2.0.0                    pypi_0    pypi
sphinxcontrib-jsmath      1.0.1                    pypi_0    pypi
sphinxcontrib-qthelp      1.0.3                    pypi_0    pypi
sphinxcontrib-serializinghtml 1.1.5                    pypi_0    pypi
sqlite                    3.36.0               hc218d9a_0  
sysroot_linux-64          2.12                he073ed8_15    conda-forge
tbb                       2021.6.0             hdb19cb5_0  
testpath                  0.5.0                    pypi_0    pypi
theano-pymc               1.1.2                    pypi_0    pypi
threadpoolctl             2.2.0                    pypi_0    pypi
tk                        8.6.10               hbc83047_0  
tktable                   2.10                 hb7b940f_3    conda-forge
tomli                     2.0.1                    pypi_0    pypi
tornado                   6.1              py39h3811e60_1    conda-forge
tqdm                      4.62.0                   pypi_0    pypi
traitlets                 5.0.5                    pypi_0    pypi
transitleastsquares       1.0.28                   pypi_0    pypi
ttvfast                   0.3.0                    pypi_0    pypi
typing-extensions         3.10.0.0                 pypi_0    pypi
tzdata                    2021e                hda174b7_0  
ultranest                 3.4.4                    pypi_0    pypi
uncertainties             3.1.6                    pypi_0    pypi
urllib3                   1.26.6                   pypi_0    pypi
wcwidth                   0.2.5                    pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
wheel                     0.36.2             pyhd3eb1b0_0  
xarray                    0.19.0                   pypi_0    pypi
xz                        5.2.5                h7b6447c_0  
zeus-mcmc                 2.4.1                    pypi_0    pypi
zipp                      3.5.0                    pypi_0    pypi
zlib                      1.2.11               h7b6447c_3  
zstd                      1.4.9                haebb681_0  
