<svelte:head>
	<title>XOR AI Tensorflow.js</title>
</svelte:head>

<script>
	import * as tf from "@tensorflow/tfjs";
	let firstNum = 0;
	let secondNum = 0;
	let trainResult = "Not trained for now";
	let testResult = "Not tested for now";
	let buttonTesting = true
	
	const model = tf.sequential();
	const opti = tf.train.sgd(0.15);

	const hiddenLayer = tf.layers.dense({
		units: 8,
		inputShape: 2,
		activation: "relu"
	});

	const outputLayer = tf.layers.dense({
		units: 1,
		activation: "relu"
	});

	model.add(hiddenLayer);
	model.add(outputLayer);
	model.compile({
		optimizer: opti,
		loss: "meanSquaredError"
	});

	const trainingInputs = tf.tensor2d([
		[0, 0],
		[1, 0],
		[0, 1],
		[1, 1]
	]);

	const trainingOutputs = tf.tensor2d([
		[0],
		[1],
		[1],
		[0]
	]);

	async function training() {
		buttonTesting = true;
		for(let i = 0; i < 100; i++) {
			let results = await model.fit(trainingInputs, trainingOutputs);
			trainResult = results.history.loss[0];
		}
		buttonTesting = false;
	}
	
	function test(x, y) {
		let arr = model.predict(tf.tensor([parseInt(x), parseInt(y)], [1,2])).toString().split(" ");
		testResult = Math.round(parseFloat(arr[5].replace(/[\[\],]/g,"")));
	}
	$: console.log(firstNum);
	$: console.log(secondNum);
</script>

<main>
	<h1>XOR AI Tensorflow.js</h1>
	<button on:click={() => training()}>Train Me</button>
	<span style="margin-bottom: 25px">Fitness : {trainResult}</span>
	<select bind:value={firstNum}>
		<option value=0>0</option>
		<option value=1>1</option>
	</select>
	<select bind:value={secondNum}>
		<option value=0>0</option>
		<option value=1>1</option>
	</select>
	<button disabled={buttonTesting} on:click={() => test(firstNum, secondNum)}>Test the Neural Network</button>
	<span>{testResult}</span>
	<div style="margin-top: 20px; text-align: center;">
		This is a simple XOR AI made with Tensorflow.js
		<br>
		If the results are wrong you can train the AI again.
		<br>
		Sometimes the training fail and give 0.5 in fitness, reload the page.
		<br>
		The (dirty) code is open source and can be found on my github page.
		<br>
		Made by <a href="https://github.com/l0wigh" target="_blank">L0Wigh</a> with <a href="https://www.tensorflow.org/js/" target="_blank">Tensorflow.js</a> and <a href="https://svelte.dev/" target="_blank">Svelte</a>
		<br>
	</div>
</main>

<style>
	h1 {
		text-align:center;
	}
	main {
		max-width: 500px;
		display: flex;
		justify-content: center;
		flex-direction: column;
		margin: 10px auto;
	}
	:global(body) {
		background: black;
		color: white;
	}
	button {
		color: white;
		background: #ff7402;
		border: none;
	}
	button:disabled {
		background: #da0000;
		color: black;
	}
</style>