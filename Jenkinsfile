@Library("jenlib")
import groovy.transform.Field


@Field String notificationChannel = "#time-to-first-spike-on-hx"

// only sent one message to mattermost
@Field Boolean SentMattermost = false;

addBuildParameter(string(name: 'chipstring', defaultValue: "random",
		         description: 'The chip on which the experiments should be executed (in the form `W66F3`). If this string is "random", a random free chip will be used.'))


def jeshWithLoggedStds(String script, String filenameStdout, String filenameStderr) {
    // pipefail to make sure 'jesh' fails if the actual script fails and it is not covered by tee
    // using 3>&1 1>&2- 2>&3- in order to switch stdout and stderr to have it both available for the notification and in the chat
    // according to https://stackoverflow.com/questions/1507816/with-bash-how-can-i-pipe-standard-error-into-another-process
    return jesh(script: "set -o pipefail; ( ( ( ${script} ) | tee ${filenameStdout} ) 3>&1 1>&2- 2>&3- ) | tee ${filenameStderr} ")
}

def recordExitSuccess(int success) {
	String filename_errors = '/jenkins/results/p_jg_FastAndDeep/execution_stats.json';

	runOnSlave(label: "frontend") {
		// add to the json file
		jesh(
			"jq --arg success '${success}' --arg BUILD_NUMBER '${env.BUILD_NUMBER}' --arg STAGE_NAME '${env.STAGE_NAME}' --arg HX 'W${wafer}F${fpga}' --arg DATE " + '"$(date +%s)" '
			+
			'\'. + {($BUILD_NUMBER): {"HX": $HX, "laststep": $STAGE_NAME, "success": $success, "timestamp": $DATE}}\' ' + "${filename_errors} > ${filename_errors}.tmp"
		);
		jesh("mv ${filename_errors}.tmp ${filename_errors}");
	}
}


def beautifulMattermostSend(Throwable t, Boolean readError) {
	if (SentMattermost) {
		throw t
	}

	String tmpErrorMsg = ""
	if(readError) {
		runOnSlave(label: "frontend") {
			// if file does not exist, create it
			jesh("[ -f tmp_stderr.log ] || touch tmp_stderr.log");
			tmpErrorMsg = readFile('tmp_stderr.log');
		}
		// too long messages lead to (cryptic) errors, so shorten the error message
		if (tmpErrorMsg.length() > 1000 ) {
		    tmpErrorMsg = "[Error was too long, check log; beginning and end are the following:]\n" + \
			    "${tmpErrorMsg[1..200]}\n[...]\n${tmpErrorMsg[-200..-1]}"
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
		endpoint: "https://chat.bioai.eu:6443/hooks/qrn4j3tx8jfe3dio6esut65tpr")
	print(message)
	SentMattermost = true
	currentBuild.result = 'FAILED'

	recordExitSuccess(0);

	throw t
}


try {
withCcache() {
withEnv(["USE_LAMBERTW_SCIPY=True"]) {
withModules(modules: ["waf", "ppu-toolchain"]) {

stage("waf setup") {
	inSingularity(app: "visionary-dls") {
		wafSetup(
			projects: ["model-hx-strobe"],
			setupOptions: "--clone-depth=1",
			// --gerrit-changes=16792
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
	try {
		onSlurmResource(partition: "jenkins",
				"cpus-per-task": 4,
				time: "1:0:0",
				mem: "24G") {
			inSingularity(app: "visionary-dls") {
				jesh("waf install")
			}
		}
	} catch (Throwable t) {
		beautifulMattermostSend(t, true);
	}
}

// globally define the two variables to not have them only in one scope
@Field int wafer = 0, fpga = 0;
stage("Checkout and determine chip") {
	runOnSlave(label: "frontend") {
		dir("fastAndDeep") {
			checkout scm
		}
		// determine the chip to be used in the following experiments
		chipstring = params.get('chipstring')
		if  (! chipstring.matches(/W[0-9]+F[0,3]/)) {
			print ("The given chip string does not match the regex /W[0-9]+F[0,3]/ using tools-slurm to find a random free chip")
			try {
				withModules(modules: ["tools-slurm"]) {
					chipstring = jesh(script: "find_free_chip.py --random",
						returnStdout: true).trim()
				}
			} catch (Throwable t) {
				print ("There is no free chip, so the default Jenkinssetup W62F0 is used")
				chipstring = "W62F0"
			}
		}
		wafer = Integer.parseInt(chipstring.split("W")[1].split("F")[0])
		fpga = Integer.parseInt(chipstring.split("W")[1].split("F")[1])
		print ("using chip ${chipstring} of wafer ${wafer} and fpga ${fpga}")
	}
}

stage("reconfigure chip") {
	try {
		onSlurmResource(partition: "cube",
				"cpus-per-task": 8,
				wafer: "${wafer}",
				"fpga-without": "${fpga}",
				time: "10:0",
				mem: "8G") {
			inSingularity(app: "visionary-dls") {
				withModules(modules: ["sw-macu_x86"]) {
					jesh("hxcube_control.py --reconfigure")
				}
			}
		}
	} catch (Throwable t) {
		beautifulMattermostSend(t, true);
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
						"cd fastAndDeep/src; python py/generate_calibration.py --output calibrations/W${wafer}F${fpga}.npz",
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
			jesh("echo 'no patch necessary anymore'")
		}
	}
}

stage("adapt hx_settings.yaml to current wafer/FPGA") {
	runOnSlave(label: "frontend") {
		dir("fastAndDeep/src/py") {
			jesh("echo 'no replacement necessary anymore'")
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
				time: "6:0:0",
				mem: "16G") {
			inSingularity(app: "visionary-dls") {
				withModules(modules: ["localdir"]) {
					// to get all information about the executing node
					jesh('env')
					jeshWithLoggedStds(
						// python path export because changed strobe interface must be loaded; USE_LAMBERTW_SCIPY because CUDA implementation is not installed
						'cd fastAndDeep/src; export PYTHONPATH="${PWD}/py:$PYTHONPATH"; USE_LAMBERTW_SCIPY=yes python experiment.py train ../experiment_configs/yin_yang_hx.yaml',
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
					jesh('env')
					jesh('[ "$(ls fastAndDeep/experiment_results/ | wc -l)" -gt 0 ] && ln -sv $(ls fastAndDeep/experiment_results/ | tail -n 1) fastAndDeep/experiment_results/lastrun')
					// runs inference for X times
					for(int i = 0;i<10;i++) {
						jeshWithLoggedStds(
							// python path export because changed strobe interface must be loaded; USE_LAMBERTW_SCIPY because CUDA implementation is not installed
							'cd fastAndDeep/src; export PYTHONPATH="${PWD}/py:$PYTHONPATH"; USE_LAMBERTW_SCIPY=yes python experiment.py inference ../experiment_results/lastrun | tee -a ../../inference.out',
							"tmp_stdout.out",
							"tmp_stderr.log"
						)
					}
				}
			}
		}
	} catch (Throwable t) {
		beautifulMattermostSend(t, true);
	}
}


stage("finalisation") {
	runOnSlave(label: "frontend") {
		archiveArtifacts 'fastAndDeep/experiment_results/lastrun/epoch_500/*.png'
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
		// write short and long term stats into a file
		inSingularity(app: "visionary-dls") {
			jesh('cd fastAndDeep/src/py; (python jenkins_executionStats.py --numberBuilds=10; echo; echo; python jenkins_executionStats.py --numberBuilds=50) > jenkinsExecutionStats.log')
			jesh('cd fastAndDeep/src/py; pango-view -qo jenkinsExecutionStats.png jenkinsExecutionStats.log')
		}
		archiveArtifacts 'fastAndDeep/src/py/jenkinsExecutionStats.*'
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
			// record that all went through
			recordExitSuccess(1);
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
  <h1>Execution stats</h1>
  <img width=450 src="lastSuccessfulBuild/artifact/fastAndDeep/src/py/jenkinsExecutionStats.png"/>
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
  <img width=300 src="lastSuccessfulBuild/artifact/fastAndDeep/experiment_results/lastrun/epoch_500/yin_yang_classification_train.png"/>
  <img width=300 src="lastSuccessfulBuild/artifact/fastAndDeep/experiment_results/lastrun/epoch_500/yin_yang_classification_test.png"/>
  <br />
  <img width=600 src="lastSuccessfulBuild/artifact/fastAndDeep/experiment_results/lastrun/epoch_500/yin_yang_summary_plot.png"/>
</p>
""")
