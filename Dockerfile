FROM python:3.7-bullseye

# Install the pandoc package and related (for automatic reporting)
RUN apt-get update -qq
RUN apt-get install -y  \
    pandoc texlive-latex-recommended texlive-latex-extra texlive-pictures

# Cleanup apt-get stuff
RUN apt-get -y autoremove &&\
    apt-get clean autoclean &&\
    rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Move sources inside the image
COPY heightmap_interpolation /usr/local/src/heightmap_interpolation/heightmap_interpolation
COPY docs /usr/local/src/heightmap_interpolation/docs
COPY Readme.md /usr/local/src/heightmap_interpolation/Readme.md
COPY LICENSE /usr/local/src/heightmap_interpolation/LICENSE
COPY MANIFEST.in /usr/local/src/heightmap_interpolation/MANIFEST.in
COPY setup.py /usr/local/src/heightmap_interpolation/setup.py

# Install the requirements and the package itself
WORKDIR /usr/local/src/heightmap_interpolation
RUN python setup.py install

# Put the dir containing the main "interpolate_netcdf4.py" in the path for convenience
ENV PATH=$PATH:/usr/local/src/heightmap_interpolation/heightmap_interpolation/apps
# And also the reporter
ENV PATH=$PATH:/usr/local/src/heightmap_interpolation/heightmap_interpolation/reporter

# Solve matplotlib complain about not finding writable /.cache/matplotlib and /.config/matplotlib folders
RUN mkdir -p /.cache/matplotlib
RUN mkdir -p /.config/matplotlib
RUN chmod -R 777 /.cache/matplotlib
RUN chmod -R 777 /.config/matplotlib

CMD ["bash"]
