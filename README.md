# GPU Tasks Scheduler

This tool is designed to manage and schedule GPU tasks, offering automated task handling and GPU resource allocation.

## Features

- **GPU Configuration**: Dynamically configure GPU settings to efficiently utilize available resources.
- **Task Scheduling**: Supports automatic task scheduling to optimize GPU usage.
- **Log Visualization**: Provides interfaces to visualize and monitor task logs in real-time.
- **Interactive Web Interface**: Facilitates task management through a user-friendly web interface, including GPU status updates, task progress, and manual control options like task termination.

## Installation

Clone the repository and install the required packages using the following commands:

```bash
git clone git@github.com:qywu/TaskScheduler.git
cd TaskScheduler
pip install -e .
```

## Configuration

Adjust the `config.yaml` file according to your system and task requirements. This configuration file controls various operational parameters such as GPU allocation, task priorities, and retry limits.


## Usage

### Starting the Service

To initialize the server that manages task scheduling and monitoring, use the command:

```bash
taskrun init
```
This command starts the web server and the task scheduler, preparing the system to accept and manage tasks.

### Submitting Tasks

Submit a task using the command line. For example, to request one GPU with 30GB of memory and a 10-second time interval between tasks, use:


```bash
taskrun -n 1 -m 30000 -t 10 python test.py
```

### Web Interface

The interface provides real-time data and control options for all scheduled tasks. You can visit `http://{config.server.host}:{config.server.port}`for logs.

## API Endpoints


The scheduler also exposes several API endpoints for programmatic control and monitoring:

* /stream_log/<job_id>: Stream logs for a specific job.
* /view_log/<job_id>: View the complete log for a specific job.
* /kill_job/<job_id>: Terminate an active job.
* /gpus_info: Retrieve information about all available GPUs.
* /update_enabled_gpus: Enable or disable specific GPUs for task scheduling.
* /cleanup: Clean up and reset task states and logs.

These endpoints facilitate integration with other systems and provide extensive control over task management and GPU scheduling.


## Contributing

Contributions to improve or enhance the GPU Tasks Scheduler are welcome. Please follow the standard procedures for submitting issues, feature requests, and pull requests.

