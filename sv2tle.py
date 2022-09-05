#!/usr/bin/env python3
import numpy as np
import yaml
from pathlib import Path

from twoline.twoline import TwoLineElement, rv2tle

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import ITRS, GCRS, TEME
from astropy.coordinates import CartesianDifferential, CartesianRepresentation

if __name__ == "__main__":
    data = yaml.safe_load(Path("./qubik2.yml").read_text())
    pitrs = np.array(data['state_vector']['r']) * u.m
    vitrs = np.array(data['state_vector']['v']) * u.m / u.s

    pitrs = CartesianRepresentation(pitrs.T)
    vitrs = CartesianDifferential(vitrs.T)

    if 'time' in data['state_vector'].keys():
        t = Time(data['state_vector']['time'], format="isot", scale="utc")
    else:
        t = Time(data['launch_time'], format="isot", scale="utc")
        t += TimeDelta(data['state_vector']['elapsed_time'] * u.s)

    pos = ITRS(pitrs.with_differentials(vitrs), obstime=t).transform_to(GCRS(obstime=t))

    p = pos.cartesian.xyz.to(u.km).value
    v = pos.velocity.d_xyz.to(u.km / u.s).value


    newtle, converged = rv2tle(data['norad_id'], t.datetime, p, v)
    print(newtle)
    print(converged)
    newtle.print_tle()
