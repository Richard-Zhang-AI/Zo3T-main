<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="description" content="A demonstration of video generation models.">
  <meta name="keywords" content="Video Generation, AI, Demo">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zo3T: Zero-shot 3D-Aware Trajectory-Guided image-to-video generation via Test-Time Training</title>

  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.3/dist/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

  <style>
    .hero.teaser {
      position: relative;
      overflow: hidden;
      min-height: 400px;
    }
    .hero.teaser video {
      position: absolute;
      top: 50%;
      left: 50%;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: translate(-50%, -50%);
      z-index: -1;
    }
    .hero-body .subtitle {
        color: white;
        text-shadow: 2px 2px 4px #000000;
    }
    .comparison-video video,
    .side-by-side video {
      width: 100%;
      height: auto;
      object-fit: cover;
    }
    .side-by-side img {
      width: 100%;
      object-fit: contain;
    }
    .card {
        background-color: #fdfdfd;
    }
  </style>

</head>
<body>

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title">Zo3T: Zero-shot 3D-Aware Trajectory-Guided image-to-video generation via Test-Time Training</h1>
          <!-- Author and affiliation info -->
          <div class="is-size-5 publication-authors" style="margin-bottom: 0.5em;">
            <span class="author-block">
              Ruicheng Zhang<sup>1,2</sup><span style="font-size:0.9em;">†</span>, Jun Zhou<sup>1</sup><span style="font-size:0.9em;">†</span>, Zunnan Xu<sup>1</sup><span style="font-size:0.9em;">†</span>, Zihao Liu<sup>1</sup>, Jiehui Huang<sup>3</sup>,<br>
              Mingyang Zhang<sup>4</sup>, Yu Sun<sup>2</sup>, Xiu Li<sup>1</sup><span style="font-size:0.9em;">*</span>
            </span>
          </div>
          <div class="is-size-6 publication-affiliations" style="color: #555;">
            <span class="affiliation-block">
              <sup>1</sup>Tsinghua University &nbsp;
              <sup>2</sup>Sun Yat-sen University<br>
              <sup>3</sup>The Hong Kong University of Science and Technology &nbsp;
              <sup>4</sup>China University of Mining and Technology
            </span>
            <br>
            <span style="font-size:0.9em;">† Co-first authors</span> &nbsp; <span style="font-size:0.9em;">* Corresponding author</span>
          </div>
        </div>
      </div>
      <!-- Abstract section -->
      <div class="columns is-centered">
        <div class="column is-three-quarters">
          <div class="box" style="background: #f7f7fa;">
            <h2 class="title is-4 has-text-centered">Abstract</h2>
            <p style="font-size: 1.1em;">
              Trajectory-Guided image-to-video (I2V) generation aims to synthesize videos that adhere to user-specified motion instructions. Existing methods typically rely on computationally expensive fine-tuning on scarce annotated datasets. Although some zero-shot methods attempt to trajectory control in the latent space, they may yield unrealistic motion by neglecting 3D perspective and creating a misalignment between the manipulated latents and the network's noise predictions. To address these challenges, we introduce Zo3T, a novel zero-shot test-time-training framework for trajectory-guided generation with three core innovations: First, we incorporate a 3D-Aware Kinematic Projection, leveraging inferring scene depth to derive perspective-correct affine transformations for target regions. Second, we introduce Trajectory-Guided Test-Time LoRA, a mechanism that dynamically injects and optimizes ephemeral LoRA adapters into the denoising network alongside the latent state. Driven by a regional feature consistency loss, this co-adaptation effectively enforces motion constraints while allowing the pre-trained model to locally adapt its internal representations to the manipulated latent, thereby ensuring generative fidelity and on-manifold adherence. Finally, we develop Guidance Field Rectification, which refines the denoising evolutionary path by optimizing the conditional guidance field through a one-step lookahead strategy, ensuring efficient generative progression towards the target trajectory.
              <br>
              Zo3T significantly enhances 3D realism and motion accuracy in trajectory-controlled I2V generation, demonstrating superior performance over existing training-based and zero-shot approaches.
            </p>
          </div>
        </div>
      </div>
      <!-- Pipeline image section -->
      <div class="columns is-centered">
  <div class="column is-full has-text-centered">
    <div class="box">
      <h2 class="title is-5">Pipeline Framework</h2>
      <figure>
        <img src="./static/images/framework.png" alt="Pipeline Framework Overview"
             style="width:100vw; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); display:block; margin:0 auto;">
        <figcaption style="margin-top:1em; font-size:1.05em; color:#444; text-align:left; padding-left:10px;">
          An overview of our zero-shot trajectory-guided video generation framework. Our method optimizes a pre-trained video diffusion model at specific denoising timesteps via two key stages. First, <b>Test-Time Training (TTT)</b> adapts the latent state and an ephemeral adapter to maintain semantic consistency along the trajectory. Second, <b>Guidance Field Rectification</b> refines the denoising direction using a one-step lookahead optimization to ensure precise path execution.
        </figcaption>
      </figure>
    </div>
  </div>
</div>
    </div>
  </div>
</section>




<section class="section">
    <div class="container is-max-desktop">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
          <h2 class="title is-3">Comparison with Other Methods</h2>
        </div>
      </div>
      
      <div class="box">
        <h3 class="title is-4 has-text-centered">Comparison Set 1</h3>
        <div class="has-text-centered" style="margin-bottom: 20px;">
            <p><strong>Condition Image</strong></p>
            <img src="./static/videos/1-compare/condition_vis.png" alt="Condition for Comparison Set 1" style="max-width: 400px; height: auto;">
        </div>
        <div class="columns is-multiline is-centered">
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">Ours</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/1-compare/Ours.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">DragAnything</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/1-compare/DragAnything.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">DragNUWA</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/1-compare/DragNUWA.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">ObjCtrl-2.5D</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/1-compare/ObjCtrl-2.5D.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">SG-I2V</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/1-compare/SG-I2V.mp4" type="video/mp4"></video>
           </div>
        </div>
      </div>

      <div class="box">
        <h3 class="title is-4 has-text-centered">Comparison Set 2</h3>
        <div class="has-text-centered" style="margin-bottom: 20px;">
            <p><strong>Condition Image</strong></p>
            <img src="./static/videos/2-compare/condition_vis.png" alt="Condition for Comparison Set 2" style="max-width: 400px; height: auto;">
        </div>
        <div class="columns is-multiline is-centered">
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">Ours</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/2-compare/Ours.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">DragAnything</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/2-compare/DragAnything.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">DragNUWA</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/2-compare/DragNUWA.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">ObjCtrl-2.5D</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/2-compare/ObjCtrl-2.5D.mp4" type="video/mp4"></video>
           </div>
           <div class="column is-one-third comparison-video">
               <h4 class="title is-5">SG-I2V</h4>
               <video autoplay controls muted loop playsinline><source src="./static/videos/2-compare/SG-I2V.mp4" type="video/mp4"></video>
           </div>
        </div>
      </div>
      
    </div>
  </section>


<section class="hero is-light is-small">
  <div class="hero-body">
    <div class="container">
      <h2 class="title is-3 has-text-centered">Side-by-Side Demos</h2>
      <div id="results-carousel" class="carousel results-carousel">
        </div>
    </div>
  </div>
</section>


<footer class="footer">
  <div class="container">
    <div class="content has-text-centered">
      <p>This website template was adapted from the <a href="https://nerfies.github.io">Nerfies</a> project page.</p>
    </div>
  </div>
</footer>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.3/dist/js/bulma-carousel.min.js"></script>
<script>
    // Dynamically generate the side-by-side demo carousel
    const demoFolders = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    const carouselContainer = document.getElementById('results-carousel');
    
    demoFolders.forEach(i => {
      const item = document.createElement('div');
      item.className = 'item';
      item.innerHTML = `
        <div class="card">
          <div class="card-content">
            <h3 class="title is-4 has-text-centered">Demo</h3>
            <div class="columns is-vcentered is-centered side-by-side">
              <div class="column">
                <p class="has-text-centered"><strong>Condition Image</strong></p>
                <img src="./static/videos/${i}/condition_vis.png" alt="Condition for Demo ${i}">
              </div>
              <div class="column">
                <p class="has-text-centered"><strong>Generated Video</strong></p>
                <video autoplay controls muted loop playsinline>
                  <source src="./static/videos/${i}/result.mp4" type="video/mp4">
                </video>
              </div>
            </div>
          </div>
        </div>
      `;
      carouselContainer.appendChild(item);
    });

    // Initialize the carousel
    bulmaCarousel.attach('#results-carousel', {
      slidesToScroll: 1,
      slidesToShow: 1, // Show one full side-by-side demo at a time
      loop: true,
      autoplay: true,
      autoplaySpeed: 5000, // Slower speed to allow for viewing
    });
</script>

</body>
</html>
