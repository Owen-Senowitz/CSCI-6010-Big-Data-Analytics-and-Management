let map;
let departureMarker = null;
let arrivalMarker = null;
let departureLatLng = null;
let arrivalLatLng = null;

function initMap() {
	// Initialize the map
	map = L.map("map").setView([40.7128, -74.006], 12); // NYC coordinates

	// Add OpenStreetMap tiles
	L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
		attribution:
			'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
	}).addTo(map);

	// Add click listener to place markers
	map.on("click", (event) => {
		const { lat, lng } = event.latlng;

		if (!departureMarker) {
			// Add departure marker
			departureLatLng = event.latlng;
			departureMarker = L.marker([lat, lng], { draggable: false }).addTo(map);
			departureMarker.bindPopup("Departure Location").openPopup();
		} else if (!arrivalMarker) {
			// Add arrival marker
			arrivalLatLng = event.latlng;
			arrivalMarker = L.marker([lat, lng], { draggable: false }).addTo(map);
			arrivalMarker.bindPopup("Arrival Location").openPopup();
		} else {
			alert("Both departure and arrival locations are already selected.");
		}
	});
}

initMap();

document
	.getElementById("predict-button")
	.addEventListener("click", async () => {
		if (!departureLatLng || !arrivalLatLng) {
			alert("Please select both departure and arrival locations on the map.");
			return;
		}

		// Prepare payload
		const isoDatetime = new Date().toISOString();
		const formattedDatetime = isoDatetime.split(".")[0];

		const payload = {
			pickup_datetime: formattedDatetime,
			pickup_latitude: departureLatLng.lat,
			pickup_longitude: departureLatLng.lng,
			dropoff_latitude: arrivalLatLng.lat,
			dropoff_longitude: arrivalLatLng.lng,
		};

		try {
			// Send prediction request
			const response = await fetch("/predict", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(payload),
			});

			const result = await response.json();

			if (result.error) {
				document.getElementById("results").innerText = `Error: ${result.error}`;
			} else {
				// Convert predictions to minutes and seconds format
				const resultText = Object.entries(result)
					.map(([model, prediction]) => {
						const minutes = Math.floor(prediction / 60);
						const seconds = Math.round(prediction % 60);
						return `${model}: ${minutes} min ${seconds} sec`;
					})
					.join("\n");
				document.getElementById("results").innerText = resultText;
			}
		} catch (error) {
			document.getElementById("results").innerText = `Error: ${error.message}`;
		}
	});

document.getElementById("fetch-results").addEventListener("click", async () => {
	try {
		const response = await fetch("results");
		const results = await response.json();

		if (results.error) {
			document.getElementById(
				"evaluation-results"
			).innerText = `Error: ${results.error}`;
		} else {
			const resultsText = Object.entries(results)
				.map(
					([model, metrics]) =>
						`${model} - Mean Squared Error: ${metrics.mean_squared_error.toFixed(
							2
						)}, R² Score: ${metrics.r2_score.toFixed(2)}`
				)
				.join("\n");
			document.getElementById("evaluation-results").innerText = resultsText;
		}
	} catch (error) {
		document.getElementById(
			"evaluation-results"
		).innerText = `Error: ${error.message}`;
	}
});
