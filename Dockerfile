# Use the official royerlab/ultrack image as the base
FROM royerlab/ultrack:0.6.1-cuda11.8

# Define an environment variable for the data directory
ENV DATA_DIR="/app/data"

# Install any utilities needed (e.g., git) to fetch a specific commit if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tini \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

# Copy everything from the Docker build context (the same directory as this Dockerfile)
# into /app inside the container
COPY . /app

# Set the working directory
WORKDIR /app

# Installing additional Python dependencies
RUN pip install --no-cache --root-user-action ignore -r requirements.txt --no-deps
RUN pip install --no-cache --root-user-action ignore ./ultrack
RUN pip install --no-cache --root-user-action ignore ./dexp-dl --no-deps

# tini is needed to run xvfb-run
# xvfb-run is needed to run GUI applications in headless mode (e.g. napari-reader)
ENTRYPOINT ["tini", "--", "xvfb-run"]

# Execute all submissions
CMD ["/usr/bin/bash", "run_all.sh"]

# If you want to run a shell instead, uncomment the following line
# CMD ["/usr/bin/bash"]
