# Use a python base image
FROM python:3.10.8 
# as build-image

# Set working directory /app dir
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy & install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install awslambdaric
RUN pip install awslambdaric

# Copy specified files with proper permissions
COPY --chown=root:root inference inference
COPY --chown=root:root modeling modeling
COPY --chown=root:root utils utils
COPY --chown=root:root config config
COPY .aws /root/.aws

# Ensure proper permissions for /app directory
RUN chmod -R 755 /app

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Pass the name of the function handler as an argument to the runtime
CMD [ "inference.inference.lambda_handler" ]
