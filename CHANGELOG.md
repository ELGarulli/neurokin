## v0.3.14
* Fix: added lower casting to string for coordinates detection from config for features calculation

## v0.3.13
* Improve: added error raise of wrong vector shape during angle calculation

## v0.3.12
* Fix: Fixed test failing due to old syntax of np.reshape

## v0.3.11
* Fix: updated the setup to reflect relaxed requirements

## v0.3.10
* Fix: Forgot to bump version so release didnt push

## v0.3.9
* Update: Relaxed requirements after c3d package update

## v0.3.8
* corrected feature datatype

## v0.3.7
* just bumping the version to force publishing
  
## v0.3.6
* Update: KinematicData configpath is now a parameter of extract_features

## v0.3.5
* Fix: deleted unnecessary casting to lowercase when importing c3d markers
* Fix: fixed import so markers id can be left as an empty list meaning all markers

## v0.3.4
* Update: exposed parameters to set step detection

## v0.3.3
* Fix: fixed plotting of the step partition which still looked like DLC

## v0.3.2

### Updates
* Fix: Fixed a bug that caused misalignment in the join and markers features
* Fix: Fixed bug from Issue #18, when get_binned was set to False while running feature extraction
* Fix: Fixed bug from Issue #19, steps partition still looks for DLC-like structures
* Update: deleted functions related to FOOOF and dropped requirement



