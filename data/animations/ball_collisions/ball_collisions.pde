/**
 * Circle Collision with Swapping Velocities
 * by Ira Greenberg. 
 * 
 * Based on Keith Peter's Solution in
 * Foundation Actionscript Animation: Making Things Move!
 */
 
int frame_rate = 1000; //<>//
boolean save_gif = true;
boolean save_frames = false;
boolean rand_background = true;
int num_frames = 100;
PImage[] images = new PImage[num_frames];

import gifAnimation.*;

GifMaker gifExport;
 
// Ball[] balls =  { 
//   new Ball(100, 400, 20), 
//   new Ball(700, 400, 80) 
// };

Ball[] balls = new Ball[2];

void setup() {
  size(50, 50); // width, height (like u,v)
  balls[0] = new Ball(width*0.25, height*1.1, (int) height*0.1); // u, v, r
  balls[1] = new Ball(width*1.1, height*1.1, (int) height*0.165);
  frameRate(frame_rate);
  if (save_gif == true) {
    gifExport = new GifMaker(this, "ball_collisions.gif");
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }

  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/50x100/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  // Set the background to random pixels or greyish:
  if (rand_background) {
  image(images[frameCount-1], 0, 0, width, height);
  } else {
    background(51);
  }

  for (Ball b : balls) {
    b.update();
    b.display();
    b.checkBoundaryCollision();
  }
  
  balls[0].checkCollision(balls[1]);
  
  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }

  if (save_frames == true) {
    saveFrame("frames/circle_vertical/###.png");
    if (frameCount == num_frames) {
      exit();
    }
  }
}
