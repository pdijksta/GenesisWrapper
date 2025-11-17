#!/bin/bash
#=======================================================================
# This example bash script:
# - calls the X0h program from https://x-server.gmca.aps.anl.gov/
# - gets the scattering factors chi_0r and chi_0i
# - saves them into a file as a function of energy
#
# The script needs the following external programs:
# 1) wget : text-based web browser
# 2) sed  : GNU stream editor
# 3) bc   : arbitrary-precision calculator language
#
# On Microsoft Windows this script can run under
# Cygwin (https://www.cygwin.com).
#-----------------------------------------------------------------------
# Version 1.0 2014/06/11 Original version: Mojmir Meduna    2003/08/27
#                        Adapted version: Sergey Stepanov   2005/12/31
#                        Non-integer math: Jackson Williams 2014/06/11
# Version 1.1 2021/06/29 Cleaned up and changed to https and curl (Sergey Stepanov)
# Version 2.0 2022/08/01 Adapted to Cloudflare and changed bc to awk
#=======================================================================

  url='https://x-server.gmca.aps.anl.gov/cgi/x0h_form.exe'

## #--------------Input parameters---------------------------
### Energy range and the number of energy points
  E1=7.7			#start energy
  E2=10			#end energy
  n=25			#number of pts (please stay within a few dozen!)

### Energy step:
# dE=$(echo "($E2-$E1)/($n-1)" | bc -l)
  dE=$(awk "BEGIN {printf \"%.3f\",($E2-$E1)/($n-1)}")

###--------------Parameters of X0h CGI interface--------------
  xway=2		# 1 - wavelength, 2 - energy, 3 - line type
# wave=$1		# works with xway=2 or xway=3
# line='Cu-Ka1'		# works with xway=3 only
  line=''       	# works with xway=3 only

### Target:
  coway=0		# 0 - crystal, 1 - other material, 2 - chemicalformula
### Crystal
  code=Diamond	# works with coway=0 only
### Other material
  amor=''               # works with coway=1 only
### Chemical formula and density (g/cm3):
  chem=''		# works with coway=2 only
  rho=''                # works with coway=2 only

### Miller indices:
  i1=3
  i2=3
  i3=1

### Output file:
  file=x0h_resultsC${i1}${i2}${i3}.dat


### Database Options for dispersion corrections df1, df2:
### -1 - Automatically choose DB for f',f"
###  0 - Use X0h data (5-25 keV or 0.5-2.5 A) -- recommended for Bragg diffraction.
###  2 - Use Henke data (0.01-30 keV or 0.41-1240 A) -- recommended for soft x-rays.
###  4 - Use Brennan-Cowan data (0.03-700 keV or 0.02-413 A)
###  6 - Use Windt data (0.01-100 KeV or 0.12-1240 A)
###  8 - Use Chantler/NIST data (0.01-450 KeV or 0.28-1240 A)
### 10 - Compare results for all of the above sources.
  df1df2=-1

  modeout=1	# 0 - html out, 1 - quasy-text out with keywords
  detail=0	# 0 - don't print coords, 1 = print coords
###-----------------------------------------------------------

  echo "Output file: $file"
###--------Print header------------------------
  msg=$(echo "#Energy,      xr0,        xi0,      xrh,      xih")
  echo $msg
  echo $msg >| $file

###--------Loop over energy--------------------
  for ((i=0; i < $n; i++)); do		# loop over energy points
#    wave=$(echo "$E1+$dE*$i" | bc -l)
     wave=$(awk "BEGIN {printf \"%.3f\",$E1+$dE*$i}")

###--------Building address--------------------
     address=$url
     address+='?xway='$xway
     address+='&wave='$wave
     address+='&line='$line
     address+='&coway='$coway
     address+='&code='$code
     address+='&amor='$amor
     address+='&chem='$chem
     address+='&rho='$rho
     address+='&i1='$i1
     address+='&i2='$i2
     address+='&i3='$i3
     address+='&df1df2='$df1df2
     address+='&modeout='$modeout
     address+='&detail='$detail

###-----------Connect & Download-----------------
### Find line with keyword, erase everything before the data, and print:
     x=( $(curl --silent -j -b name=cookies $address | sed -n '/xr0=/{s/.*xr0=//;p;};/xi0=/{s/.*xi0=//;p;};/xrhsg=/{s/.*xrhsg=//;p;};/xihsg=/{s/.*xihsg=//;p;}') )

###--------Print current point-------------------
	 msg=$(echo "   ${wave},  ${x[0]},  ${x[1]},   ${x[2]},    ${x[3]}")
     echo $msg
     echo $msg >> $file
  done


