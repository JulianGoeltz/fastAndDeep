@Library("jenlib")
import groovy.transform.Field

@Field String notificationChannel = "#hicann-dls-users"

addBuildParameter(string(name: 'chipstring', defaultValue: "random",
		         description: 'The chip on which the experiments should be executed (in the form `W66F3`). If this string is "random", a random free chip will be used.'))


def jeshWithLoggedStds(String script, String filenameStdout, String filenameStderr) {
    // pipefail to make sure 'jesh' fails if the actual script fails and it is not covered by tee
    // using 3>&1 1>&2- 2>&3- in order to switch stdout and stderr to have it both available for the notification and in the chat
    // according to https://stackoverflow.com/questions/1507816/with-bash-how-can-i-pipe-standard-error-into-another-process
    return jesh(script: "set -o pipefail; ( ( ( ${script} ) | tee ${filenameStdout} ) 3>&1 1>&2- 2>&3- ) | tee ${filenameStderr} ")
}


def beautifulMattermostSend(Throwable t, Boolean readError) {
	String tmpErrorMsg = ""
	if(readError) {
		runOnSlave(label: "frontend") {
			tmpErrorMsg = readFile('tmp_stderr.log')
		}
		if ( tmpErrorMsg != "" ) {
			tmpErrorMsg = "\n\n```\n${tmpErrorMsg}\n```"
		}
	}
	String message = "Jenkins build [`${env.JOB_NAME}/${env.BUILD_NUMBER}`](${env.BUILD_URL}) failed at `${env.STAGE_NAME}` on `W${wafer}F${fpga}`!\n```\n${t.toString()}\n```${tmpErrorMsg}"
	mattermostSend(
		channel: notificationChannel,
		message: message,
		failOnError: true,
		endpoint: "https://chat.bioai.eu/hooks/qrn4j3tx8jfe3dio6esut65tpr")
	print(message)
	currentBuild.result = 'FAILED'
}


try {
withCcache() {
withEnv(["USE_LAMBERTW_SCIPY=True"]) {
withModules(modules: ["waf", "ppu-toolchain"]) {

stage("waf setup") {
	inSingularity(app: "visionary-dls") {
		wafSetup(
			projects: ["model-hx-strobe"],
			setupOptions: "--clone-depth=1 --gerrit-changes=13234,15180,15691",
			noExtraStage: true
		)
	}
}

stage("waf configure") {
	onSlurmResource(partition: "jenkins",
			"cpus-per-task": 8,
			time: "5:0") {
		inSingularity(app: "visionary-dls") {
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

// globally define the two variables to not have them only in one scope
@Field int wafer = 0, fpga = 0;
stage("Checkout and determine chip") {
	// for sure done wrong, but how to put it into a subfolder and special branch with scm?
	runOnSlave(label: "frontend") {
		jesh("git clone -b feature/JenkinsjobUpdates https://github.com/JulianGoeltz/fastAndDeep")
		// determine the chip to be used in the following experiments
		chipstring = params.get('chipstring')
		if  (! chipstring.matches(/W[0-9]+F[0,3]/)) {
			print ("The given chip string does not match the regex /W[0-9]+F[0,3]/ using tools-slurm to find a random free chip")
			withModules(modules: ["tools-slurm"]) {
				chipstring = jesh(script: "find_free_chip.py --random",
					returnStdout: true).trim()
			}
		}
		wafer = Integer.parseInt(chipstring.split("W")[1].split("F")[0])
		fpga = Integer.parseInt(chipstring.split("W")[1].split("F")[1])
		print ("using chip ${chipstring} of wafer ${wafer} and fpga ${fpga}")
	}
}

stage("create calib") {
	try {
		onSlurmResource(partition: "cube",
				"cpus-per-task": 8,
				wafer: "${wafer}",
				"fpga-without": "${fpga}",
				time: "10:0",
				mem: "8G") {
			inSingularity(app: "visionary-dls") {
				withModules(modules: ["localdir"]) {
					jesh("module list")
					jesh("module show localdir")
					jeshWithLoggedStds(
						"cd model-hx-strobe/experiments/yinyang; python generate_calibration.py --output ../../../fastAndDeep/src/calibrations/tmp_jenkins.npz",
						"tmp_stdout.log",
						"tmp_stderr.log"
					)
				}
			}
		}
	} catch (Throwable t) {
		beautifulMattermostSend(t, true);
	}
}

stage("patch strobe backend") {
	runOnSlave(label: "frontend") {
		dir("fastAndDeep/src") {
			jesh("patch ../../lib/strobe/backend.py -i py/libStrobeBackend.patch")
		}
	}
}

stage("adapt hx_settings.yaml to current wafer/FPGA") {
	runOnSlave(label: "frontend") {
		dir("fastAndDeep/src/py") {
			jesh("sed -i 's/temp_for_jenkins/W${wafer}F${fpga}/' hx_settings.yaml")
		}
	}
}

stage("get datasets") {
	runOnSlave(label: "frontend") {
		inSingularity(app: "visionary-dls") {
			dir("fastAndDeep/src") {
				// downloads MNIST (not used at the moment)
				// jesh("python -c \"import torchvision; print(torchvision.datasets.MNIST('../data/mnist', train=True, download=True))\"")
				// downloads YY set
				jesh("git submodule update --init")
			}
		}
	}
}

stage("training") {
	//// potentially shorten the training, for testing purposes
	//runOnSlave(label: "frontend") {
	//	dir("fastAndDeep") {
	//		jesh("sed -i 's/epoch_number: 300/epoch_number: 10/' experiment_configs/yin_yang_hx.yaml")
	//		jesh("sed -i 's/epoch_snapshots: \\[1, 5, 10, 15, 50, 100, 150, 200, 300\\]/epoch_snapshots: \\[1 \\]/' experiment_configs/yin_yang_hx.yaml")
	//		jesh("sed -i 's/^    write_new_data/    # write_new_data()/' src/py/jenkins_elastic.py")
	//	}
	//}
	try {
		onSlurmResource(partition: "cube",
				"cpus-per-task": 8,
				wafer: "${wafer}",
				"fpga-without": "${fpga}",
				time: "5:0:0",
				mem: "16G") {
			inSingularity(app: "visionary-dls") {
				withModules(modules: ["localdir"]) {
					// to get all information about the executing node
					jesh('env')
					jeshWithLoggedStds(
						'cd fastAndDeep/src; export PYTHONPATH="${PWD}/py:$PYTHONPATH"; python experiment.py train ../experiment_configs/yin_yang_hx.yaml',
						"tmp_stdout.log",
						"tmp_stderr.log"
					)
				}
			}
		}
	} catch (Throwable t) {
		beautifulMattermostSend(t, true);
	}
}

stage("inference") {
	try {
		onSlurmResource(partition: "cube",
				"cpus-per-task": 8,
				wafer: "${wafer}",
				"fpga-without": "${fpga}",
				time: "10:0",
				mem: "16G") {
			inSingularity(app: "visionary-dls") {
				withModules(modules: ["localdir"]) {
					jesh('[ "$(ls fastAndDeep/experiment_results/ | wc -l)" -gt 0 ] && ln -sv $(ls fastAndDeep/experiment_results/ | tail -n 1) fastAndDeep/experiment_results/lastrun')
					// runs inference for X times
					jeshWithLoggedStds(
						'cd fastAndDeep/src; export PYTHONPATH="${PWD}/py:$PYTHONPATH"; for i in $(seq 10); do python experiment.py inference ../experiment_results/lastrun; done',
						"inference.out",
						"tmp_stderr.log"
					)
				}
			}
		}
	} catch (Throwable t) {
		beautifulMattermostSend(t, true);
	}
}


stage("finalisation") {
	runOnSlave(label: "frontend") {
		archiveArtifacts 'fastAndDeep/experiment_results/lastrun/epoch_300/*.png'
		archiveArtifacts 'fastAndDeep/src/live_accuracy.png'
		// plot short detailed summary
		inSingularity(app: "visionary-dls") {
			jesh('cd fastAndDeep/src/py; python jenkins_elastic.py')
		}
		archiveArtifacts 'fastAndDeep/src/py/jenkinssummary_yin_yang.png'
		// plot long overview summary
		inSingularity(app: "visionary-dls") {
			jesh('cd fastAndDeep/src/py; python jenkins_elastic.py  --filename="jenkinssummary_{dataset}_longNolegend.png" --firstBuild=50 --nolegend --reduced_xticks')
		}
		archiveArtifacts 'fastAndDeep/src/py/jenkinssummary_yin_yang_longNolegend.png'
		// test whether accuracy is too low
		try {
			inSingularity(app: "visionary-dls") {
				// gets the mean of all accuracies and compares it with hard coded 92
				jeshWithLoggedStds(
					'acc=$(grep -oP "the accuracy is \\K[0-9.]*" inference.out | jq -s add/length); (( $(echo "92 > $acc" | bc -l) )) && (echo "accuracy $acc is too bad" >&2) && exit 1 || exit 0',
					"tmp_stdout.out",
					"tmp_stderr.log"
				)
			}
		} catch (Throwable t) {
			beautifulMattermostSend(t, true);
		}
	}
}


}
}
}

} catch (Throwable t) {
	beautifulMattermostSend(t, false);
}


/**
 * Setting the description of the jenkins job (to have it in the repository).
 */
setJobDescription("""
<p>
  Repository is located <a href="https://github.com/JulianGoeltz/fastanddeep">on GitHub</a>, Jenkinsjob is executed daily in the hour after midnight and should take around 1.5 hours.
  Details on the theory <a href="https://arxiv.org/abs/1912.11443">can be found in the publication arXiv:1912.11443</a>, it is used to classify <a href="https://arxiv.org/abs/2102.08211">the Yin-Yang dataset</a>.
</p>
<p>
  <h1>Summary of the last few runs</h1>
  <img width=600 src="lastSuccessfulBuild/artifact/fastAndDeep/src/py/jenkinssummary_yin_yang.png"/>
</p>
<p>
  <h1>Overview over a longer time (50 runs)</h1>
  <img width=600 src="lastSuccessfulBuild/artifact/fastAndDeep/src/py/jenkinssummary_yin_yang_longNolegend.png"/>
</p>
<p>
  <h1>Stats of last stable run</h1>
  <img width=300 src="lastSuccessfulBuild/artifact/fastAndDeep/experiment_results/lastrun/epoch_300/yin_yang_classification_train.png"/>
  <img width=300 src="lastSuccessfulBuild/artifact/fastAndDeep/experiment_results/lastrun/epoch_300/yin_yang_classification_test.png"/>
  <br />
  <img width=600 src="lastSuccessfulBuild/artifact/fastAndDeep/experiment_results/lastrun/epoch_300/yin_yang_summary_plot.png"/>
</p>
""")
