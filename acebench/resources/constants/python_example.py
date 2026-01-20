# Constants - Testing Commands
TEST_PYTEST = "pytest -rA --color=no"
TEST_PYTEST_VERBOSE = "pytest -rA --tb=short --color=no"
TEST_ASTROPY_PYTEST = "pytest -rA -vv -o console_output_style=classic --tb=no "
TEST_DJANGO = "python tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 "
TEST_DJANGO_NO_PARALLEL = "python tests/runtests.py --verbosity 2 "
TEST_SEABORN_VERBOSE = "pytest -rA --tb=long "
TEST_SPHINX = "tox --current-env -epy39 -v "
TEST_SYMPY_VERBOSE = "bin/test -C --verbose "

# Constants for test discovery commands
TEST_DISCOVERY_DEFAULT = ["python", "-m", "pytest", "--rootdir=.", "--collect-only", "-q", "--tb=no", "--continue-on-collection-errors"]

# Constants for dynamic tracing commands
TEST_DYNAMIC_TRACE_DEFAULT = "-p no:xdist --no-header --tb=short --color=no -vv -rA"
TEST_DYNAMIC_TRACE_NERF = "-n 0 --no-header --no-header --color=no --tb=short -vv -rA"

# Constants - Installation Specifications (name must match last folder of project_path, uppercase, hyphen to underscore)
SPECS_ACCELERATE = {
    # repo config
    "repository": "huggingface/accelerate",
    "commit": "a73fd3a",
    "clone_method": "https",
    "base_url": None,

    # image build config
    "base_image": "python312_cu121_torch28",
    "rebuild_base_image": False,
    "rebuild_instance_image": False,
    "custom_instance_image_build": [    # custom commands when building instance image (persisted)
    ],
    "pre_install": [    # commands before project install (not persisted)
    ],
    "install": "pip install -e .[dev,deepspeed]",   # project install command
    "pip_packages": [   # extra packages to install
    ],
    "docker_specs": {   # Docker runtime config
        "run_args": {
            "cuda_visible_devices": "4,5,6,7", # GPU config: "all" uses all GPUs, "0,1,2" uses first 3, None disables
            "number_once": 2,
            "shm_size": None,
            "cap_add": [],
        },
        "custom_docker_args": [   # custom Docker container runtime args
        ]
    },

    # test discovery config
    "test_scanner_cmd": TEST_DISCOVERY_DEFAULT,
    "timeout_scanner": 300,
    "scan_cache": True,
    "start_time": None,
    "min_test_num": 1,     # minimum test points per data item
    "max_f2p_num": -1,    # max F2P items for this repo (-1 means no limit)
    "max_p2p_num": -1,    # max P2P items for this repo (-1 means no limit)
    
    # test execution config
    "test_cmd": TEST_PYTEST_VERBOSE,
    "timeout_run": 600,
    "timeout_one": 10,
    "test_cache": True,
    
    # dynamic config
    "test_dynamic_cmd": TEST_DYNAMIC_TRACE_DEFAULT,
    "timeout_dynamic": 600,
    "dynamic_cache": True,
    
    # top selection config
    "llm_cache": True,
    "batchsize_top": 5,    # items per batch for top selection
    "max_depth_top": 5,    # max depth for top selection
    "min_p2p_files": 1,    # min p2p test files per item
    "max_p2p_files": 5,    # max p2p test files per item

    # p2p selection config
    "p2p_cache": True,
    "max_code_line_lower_bound": 3000,
    "max_code_line_upper_bound": 5000,

    # data pipeline config
    "data_cache": True,
    "timeout_data": 1200,              # max processing time per item in data stage (seconds)
    "timeout_collect": 300,            # pytest --collect-only timeout (seconds)
    "f2p_pass_rate_threshold": 0.3,    # f2p post-validation pass threshold
    "llm_prompt_for_case": True,       # enable LLM prompt generation in converter stage
    
    # misc
    "library_name": "accelerate",
    "black_links": [
        "https://github.com/huggingface/accelerate"
    ]
}