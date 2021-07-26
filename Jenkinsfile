@Library("jenlib") _


withCcache() {
withModules(modules: ["waf", "ppu-toolchain"]) {

stage("waf setup") {
	inSingularity(app: "visionary-dls") {
		wafSetup(
			//container: [app: "visionary-dls"],
			projects: ["model-hx-strobe"],
			setupOptions: "--clone-depth=1 --gerrit-changes=13293,15180",
			noExtraStage: true
			// notificationChannel: "#time-to-first-spike-on-hx"
		)
	}
}

stage("waf configure") {
	onSlurmResource(partition: "compile",
			"cpus-per-task": 8,
			time: "15:0") {
		inSingularity(app: "visionary-dls") {
			jesh("echo $SINGULARITY_CONTAINER")
			jesh("waf configure")
		}
	}
}

stage("waf install") {
	onSlurmResource(partition: "compile",
			"cpus-per-task": 16,
			time: "6:0:0",
			mem: "16G") {
		inSingularity(app: "visionary-dls") {
			jesh("waf install")
		}
	}
}

stage("Checkout") {
	// for sure done wrong, but how to put it into a subfolder with scm?
	runOnSlave(label: "frontend") {
		jesh("git clone -b feature/Jenkinsjob https://github.com/JulianGoeltz/fastAndDeep")
	}
}

stage("create calib") {
	onSlurmResource(partition: "cube",
			"cpus-per-task": 8,
			wafer: 67,
			"fpga-without": 3,
			time: "1:0:0",
			mem: "16G") {
		inSingularity(app: "visionary-dls") {
			withModules(modules: ["localdir"]) {
				jesh("module list")
				jesh("module show localdir")
				jesh("cd model-hx-strobe/experiments/yinyang; python generate_calibration.py --output ../../../fastAndDeep/src/calibrations/tmp_W67F3.npz")
			}
		}
	}
}

stage("patch strobe backend") {
	runOnSlave(label: "frontend") {
		dir("fastAndDeep/src") {
			jesh("patch ../../lib/strobe/backend.py -i py/libStrobeBackend.patch")
		}
	}
}

stage("get datasets") {
	runOnSlave(label: "frontend") {
		inSingularity(app: "visionary-dls") {
			dir("fastAndDeep/src") {
				jesh("python -c \"import torchvision; print(torchvision.datasets.MNIST('../data/mnist', train=True, download=True))\"")
				jesh("git submodule update --init")
			}
		}
	}
}

stage("training") {
	onSlurmResource(partition: "cube",
			"cpus-per-task": 8,
			wafer: 67,
			"fpga-without": 3,
			time: "5:0:0",
			mem: "16G") {
		inSingularity(app: "visionary-dls") {
			withModules(modules: ["localdir"]) {
				jesh('cd fastAndDeep/src; export PYTHONPATH="${PWD}/py:$PYTHONPATH"; python experiment.py train ../experiment_configs/yin_yang_hx.yaml')
			}
		}
	}
}

stage("inference") {
}


stage("finalisation") {
	runOnSlave(label: "frontend") {
		jesh("ls")
		jesh("pwd")
		jesh("echo $SINGULARITY_CONTAINER")
	}
}

}}
