# Start with Python base image, as BentoML requires Python
FROM python:3.9

# Create a user and set up environment for a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code into the container
COPY --chown=user . /app

# Expose the port you want your app to be accessible on (BentoML default is 3000)
EXPOSE 3000

# Run the BentoML server, replacing `<YOUR_BENTO_SERVICE_NAME>` with your actual service name
CMD ["bentoml", "serve", "co2_emission_service", "--host", "0.0.0.0", "--port", "3001"]
