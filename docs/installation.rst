===============
Getting Started
===============
To run the snow mapping tool snowicesat, we first need to clone the github repository into the desired folder by running::

   git clone <url> 

Then change into the repository and create a virtual environment with::

   conda env create -f requirements-py36-snowicesat.yml

(The package availability and dependencies were tested in Python 3.6 on Linux
and Windows)

The Crampon Module that snowicesat is based on is still being developed, we therefore have to clone and 
install the repository manually by running::
    git clone <url>
    cd crampon
    pip install -e.

Now the virtual environment is created. 
We also need to create an account for 
the Copernicus Hub that provides the Sentinel-2 Imagery.

.. _Copernicus Open Access Hub: https://scihub.copernicus.eu/dhus/#self-registration

The username and password then need to be stored in an snowicesat.credentials file as::
    ['sentinel'] 
            'user' = 'username'
            'password' = '********'

(make sure this file included in the .gitignore when pushing back to the remote server).

We can now start to setup a first run.
In the snowicesat_params.cfg file, we find the configurations for our setup.
We can define the working directory, the filepath to the DEMs that we use as an input,
the date or time frame of interest and the cloud cover range for which we want to download the data.
The further paramaters are inherited from OGGM (the Open Global Glacier Model) so details about them can be found in the OGGM  documentation.

In the setup.py file, we 

