# neurokin
correlation between neural and kinematics, refactored and expansion of visualization and analysis methods. kinematic analysis based on a modified version of GGait.


## Gait with ggait
# analyise c3d files

get the minimal ggait version
# connect matlab and python
Windows

open MATLAB and type matlabroot. From the python terminal:

cd to matlabroot\extern\engines\python

python setup.py install

NOTE: if you get error: could not create 'dist\matlabengineforpython.egg-info': Access is denied

runn your IDE as administrato

from the MATLAB command prompt:

cd (fullfile(matlabroot,'extern','engines','python'))

system('python setup.py install')

matlab.engine.shareEngine
