# Quick Guide: Building and Pushing Docker Image with Parquet File

## Prerequisites
- Docker installed and running
- Docker Hub account
- Project structure set up with Dockerfile in `docker/Dockerfile.data_image`

## Steps

1. **Copy Parquet file**
   Ensure `enveda_library_subset.parquet` is in `data/raw/` directory.

   ```bash
   cp path/to/enveda_library_subset.parquet data/raw/
   ```

2. **Build Docker image**
   From the project root:

   ```bash
   docker buildx build --platform linux/amd64 -t ghiret/hackathon_bio:latest -f docker/Dockerfile.data_image .
   ```

3. **Tag image with version** (optional)

   ```bash
   docker tag ghiret/hackathon_bio:latest ghiret/hackathon_bio:v1.0.0
   ```

4. **Push to Docker Hub**

   ```bash
   docker push ghiret/hackathon_bio:latest
   docker push ghiret/hackathon_bio:v1.0.0  # If tagged
   ```

Remember to increment the version number for each new build if using versioned tags.
