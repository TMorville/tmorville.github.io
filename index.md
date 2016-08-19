---
layout: default
author_profile: false
---

<div class="video_container">
  <video controls="controls" allowfullscreen="true" poster="/images/firing.gif">
    <source src="https://gfycat.com/DistinctSneakyFruitfly" type="video/mp4">
  </video> 
</div>

.video_container {
	position: relative;
	padding-bottom: 56.25%; /* 16:9 */
	padding-top: 25px;
	height: 0;
}

.video_container video {
	position: absolute;
	top: 0;
	left: 0;
	width: 100%;
	height: 100%;
}