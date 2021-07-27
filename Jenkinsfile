@Library("jenlib") _



try {
timeout(time: 180, unit: "MINUTES") {
withCcache() {
withModules(modules: ["waf", "ppu-toolchain"]) {

stage("waf setup") {
	inSingularity(app: "visionary-dls") {
		wafSetup(
			projects: ["model-hx-strobe"],
			setupOptions: "--clone-depth=1 --gerrit-changes=13293,15180",
			noExtraStage: true
		)
	}
}

stage("waf configure") {
	onSlurmResource(partition: "jenkins",
			"cpus-per-task": 8,
			time: "5:0") {
		inSingularity(app: "visionary-dls") {
			jesh("echo $SINGULARITY_CONTAINER")
			jesh("waf configure")
		}
	}
}

stage("waf install") {
	onSlurmResource(partition: "jenkins",
			"cpus-per-task": 4,
			time: "1:0:0",
			mem: "24G") {
		inSingularity(app: "visionary-dls") {
			jesh("waf install")
		}
	}
}

stage("Checkout") {
	// for sure done wrong, but how to put it into a subfolder and special branch with scm?
	runOnSlave(label: "frontend") {
		jesh("git clone -b feature/Jenkinsjob https://github.com/JulianGoeltz/fastAndDeep")
	}
}

stage("create calib") {
	onSlurmResource(partition: "jenkins",
			"cpus-per-task": 8,
			wafer: 67,
			"fpga-without": 3,
			time: "10:0",
			mem: "8G") {
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
	onSlurmResource(partition: "jenkins",
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
	onSlurmResource(partition: "jenkins",
			"cpus-per-task": 8,
			wafer: 67,
			"fpga-without": 3,
			time: "10:0",
			mem: "16G") {
		inSingularity(app: "visionary-dls") {
			withModules(modules: ["localdir"]) {
				jesh('[ "$(ls fastAndDeep/experiment_results/yin_yang_hx_* | wc -l)" -gt 0 ] && ln -sv $(ls fastAndDeep/experiment_results/yin_yang_hx_* | tail -n 1) fastAndDeep/experiment_results/lastrun')
				jesh('ls -lAh fastAndDeep')
				jesh('ls -lAh fastAndDeep/experiment_results')
				jesh('ls -lAh fastAndDeep/experiment_results/lastrun/')
				jesh('ls -lAh fastAndDeep/experiment_results/lastrun/epoch_300')
				jesh('ls -lAh fastAndDeep/src')
				jesh('cd fastAndDeep/src; export PYTHONPATH="${PWD}/py:$PYTHONPATH"; python experiment.py inference ../experiment_results/lastrun')
			}
		}
	}
}


stage("finalisation") {
	runOnSlave(label: "frontend") {
		archiveArtifacts 'fastAndDeep/experiment_results/lastrun/epoch_300/*.png'
		archiveArtifacts 'fastAndDeep/src/live_accuracy.png'
	}
}


}
}
}

} catch (Throwable t) {
	notifyFailure(mattermostChannel: "#time-to-first-spike-on-hx")
	throw t
}

if (currentBuild.currentResult != "SUCCESS") {
	notifyFailure(mattermostChannel: "#time-to-first-spike-on-hx")
}
