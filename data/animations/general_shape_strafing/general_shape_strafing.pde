float window_scale = 1.0;
float posX;
float posY;
float theta = 0; // Rotation angle of the shape
float gamma = 0; // Rotation angle of the motion
String shape = "rectangle"; // Shape to be drawn, can be "cross", "rectangle", or "ellipse"
String mot_dir; // Motion direction, filled in automatically based on shape 
float traj_len; // Length of the trajectory
float s_len, s_wid, old_s_len; // length and width of the shape
float thickness = 3; // stroke thickness of the shape
color e_color = color(0, 0, 0); // Color of the shape
float speed; // Speed of vertical movement
int frame_rate; //<>//
int num_frames;
PImage[] images;
float d, s_wid_divisor; // Random size value
float a, b, c; // Random color values
float thk; // Random thickness value
float th; // Random rotation value
int num_occlusions = 6;
float[] occlusionX = new float[num_occlusions];
float[] occlusionY = new float[num_occlusions];
float[] occlusionWidth = new float[num_occlusions];
float[] occlusionHeight = new float[num_occlusions];
color[] occ_colors = new color[num_occlusions];
float[] occ_rot = new float[num_occlusions];
int randomizationRate;
int ws = 50;
boolean rand_size = true;
boolean rand_color = true;
boolean rand_thickness = true;
boolean rand_rotation = true;
boolean rand_occlusions = true;
String rand_background = "pixels"; // can be "pixels", "whole", "white"

boolean save_gif = false; // only set save_gif or save_frames to true, not both, or both to false
boolean save_frames = false;

boolean train_mode = true; // just flip this one to switch between train and test modes
boolean test_mode = !train_mode;

boolean exec_randomize = true;
boolean flip = false;

import gifAnimation.*;

GifMaker gifExport;

public void settings() {
  if ((save_gif == false) && (save_frames == false)) {size(500, 500);} // Set the size of the window, w, h
  else {size(ws, ws);} // Set the size of the window, w, h
}

void setup() {
  // Set the frame rate and the number of frames to be saved per mode
  if (save_gif && save_frames) {println("Error: save_gif and save_frames cannot both be true."); exit();}
  if (save_gif == true) {num_frames = 150; frame_rate = 1000;}
  else if (save_frames == true) {num_frames = 50000; frame_rate = 50000;} // deleteDirectory(new File(save_dir));}
  else {num_frames = 1000; frame_rate = 2;}
  images = new PImage[num_frames];
  
  s_len = sqrt(2) * height / 3; // Initial length of the cross
  s_wid = s_len / 4; // Initial thickness of the cross
  speed = width / 8; // Initial speed of vertical movement
  randomizationRate = (int) (height / speed);
  posX = 4 * width / 8; // Initial X position
  posY = 4 * height / 8; // Initial Y position
  fill(e_color); // Set the inital fill color to black
  stroke(0); // Set the initial stroke color to white
  strokeWeight(thickness);  // Set the initial stroke thickness
  rectMode(CENTER);

  // Set motion direction
  if (shape == "cross") {
    gamma = 0;
    mot_dir = "R";
  } else if (shape == "rectangle") {
    gamma = -PI/4;
    mot_dir = "RD";
  } else if (shape == "ellipse") {
    gamma = -PI/2;
    mot_dir = "D";
  } else {
    println("Error: shape not defined.");
  }
  // define length of trajectory as a function of window size, ws, and motion direction, gamma
  traj_len = sqrt(width*width + height*height) * cos(PI/4 + gamma);
  
  frameRate(frame_rate);
  if (save_gif == true) {
    if (train_mode) {gifExport = new GifMaker(this, "general_" + shape + "_" + mot_dir + "_train.gif");}
    else if (test_mode) {gifExport = new GifMaker(this, "general_" + shape + "_" + mot_dir + "_test.gif");}
    else {println("Error: train_mode and test_mode not defined."); exit();}
    gifExport.setRepeat(0); // make it an "endless" animation
    gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  for (int i = 0; i < num_frames; i++) {
    images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/"+nf(ws)+"x"+nf(ws)+"/" + nf(i%1000, 3) + ".png");
  }
}

void draw() {
  // establish if randomization should be executed this frame
  // if (frameCount % randomizationRate == 1) {
  //   exec_randomize = true;
  // } else {
  //   exec_randomize = false;
  // }
  // float epsilon = 0.05 * width;
  if (frameCount == 1) {
    exec_randomize = true;
    }
    // } else if ((posY + epsilon) < height) {
    //   exec_randomize = false;
    //   flip = true;
    // } else if (flip == true) {//if (((posY + epsilon) >= height / 2) && (flip == true)){
    //   exec_randomize = true;
    //   flip = false;
    // }

  // Set the background to random pixels or white:
  if (rand_background == "pixels") {
    image(images[frameCount-1], 0, 0, width, height);
  } else if (rand_background == "whole") {
    background(color(random(100, 200), random(100, 200), random(100, 200)));
  } else if (rand_background == "white") {
    background(255);
  }

  // Update position
  posX += speed * cos(abs(gamma)); // Move the shape horizontally
  posY += speed * sin(abs(gamma)); // Move the shape vertically

  if ((posX * cos(abs(gamma)) + posY * sin(abs(gamma))) - (s_len / 2) > traj_len) { // Reset position when it goes beyond the screen
    old_s_len = s_len;
    flip = true;
    exec_randomize = true;
  }

  // Adjust size randomly
  if (rand_size && exec_randomize) {
    // original min = height/3 = 16.7, max = height/1.2 = 41.7
    // new min = 15, max = 45, skip 25 - 35
    // leave out test sizes == 27-33
    d = random(15, 40);
    if (train_mode == true) {
      while (d >= 22 && d <= 25) {d = random(15, 40);}
    } else if (test_mode == true) {
      while (d < 23 || d > 24) {d = random(23, 24);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    s_wid_divisor = random(2, 4);
    if (train_mode == true) {
      while (s_wid_divisor >= 3 && s_wid_divisor <= 3.75) {s_wid_divisor = random(2, 4);}
    } else if (test_mode == true) {
      while (s_wid_divisor < 3.25 || s_wid_divisor > 3.5) {s_wid_divisor = random(3.25, 3.5);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    if ((shape == "cross") || (shape == "rectangle")) {s_wid_divisor = s_wid_divisor * 1.5;} 
    // for debug with window_size != 50
    if (width != 50.0) {d = d * (width / 50.0);}
    s_len = d;
    s_wid = s_len / s_wid_divisor;
  }

  // Adjust color randomly
  if (rand_color && exec_randomize) {
    // original color = random(255) x 3
    // new color = random(255) x 3, if val between 100 and 200, randomize again
    // leave out test colors == random(255) x 3, if val NOT between 120 and 180, randomize again
    a = random(255);
    b = random(255);
    c = random(255);
    // if (train_mode == true) {
    //   while (a >= 100 && a <= 200) {a = random(255);}
    //   while (b >= 100 && b <= 200) {b = random(255);}
    //   while (c >= 100 && c <= 200) {c = random(255);}
    // } else if (test_mode == true) {
    //   while (a < 120 || a > 180) {a = random(120, 180);}
    //   while (b < 120 || b > 180) {b = random(120, 180);}
    //   while (c < 120 || c > 180) {c = random(120, 180);}
    // } else {
    //   println("Error: train_mode and test_mode not defined."); exit();
    // }
    e_color = color(a, b, c);
    fill(e_color);
  }

  // Adjust rotation randomly
  if (rand_rotation && exec_randomize) {
    // original theta = random(0, PI/2) aka 8*PI/16
    // new theta = random(0, PI/2), if val between 3*PI/16 and 5*PI/16, randomize again
    // leave out test rotations == random(0, PI), if val NOT between 3*PI/16 and 5*PI/16, randomize again
    th = random(PI/2);
    // if (train_mode == true) {
    //   while (th >= 3*PI/16 && th <= 5*PI/16) {th = random(PI/2);}
    // } else if (test_mode == true) {
    //   while (th < 3*PI/16 || th > 5*PI/16) {th = random(3*PI/16, 5*PI/16);}
    // } else {
    //   println("Error: train_mode and test_mode not defined."); exit();
    // }
    theta = th;
  }

  // Adjust thickness randomly
  if (rand_thickness && exec_randomize) {
    // original thickness = random(1, 5)
    // new thickness = random(1, 5), if val between 2.5 and 3.5, randomize again
    // leave out test thicknesses == random(1, 5), if val NOT between 2.5 and 3.5, randomize again
    thk = random(1, s_wid/3);
    // if (train_mode == true) {
    //   while (thk >= 2.5 && thk <= 3.5) {thk = random(1, 5);}
    // } else if (test_mode == true) {
    //   while (thk < 2.5 || thk > 3.5) {thk = random(2.5, 3.5);}
    // } else {
    //   println("Error: train_mode and test_mode not defined."); exit();
    // }
    thickness = thk;
    strokeWeight(thickness);
  }

  if (exec_randomize) {
    // reset randomizer so it is only used once per cycle
    exec_randomize = false;
  }

  // reset position when shape goes beyond the screen, accounting for the shape in shape size
  if (flip) { // Reset position when it goes beyond the screen
    posX = posX + (s_len - old_s_len) * cos(abs(gamma));
    posY = posY + (s_len - old_s_len) * sin(abs(gamma));
    posX = posX - (posX + (s_len / 2)) * cos(abs(gamma)) * sqrt(2) * cos(PI/4 + gamma);
    posY = posY - (posY + (s_len / 2)) * sin(abs(gamma)) * sqrt(2) * cos(PI/4 + gamma);
    flip = false;
  }

  pushMatrix(); // Save the current drawing style settings and transformations
  translate(posX, posY); // Move the origin to the new position
  rotate(theta); // Rotate the cross

  if (shape == "cross") {
    // Draw the cross lines
    rect(0, 0, s_wid, s_len); // Vertical line
    rect(0, 0, s_len, s_wid); // Horizontal line
  } else if (shape == "rectangle") {
    // Draw the rectangle
    rect(0, 0, s_len, s_wid);
  } else if (shape == "ellipse") {
    // Draw the ellipse
    ellipse(0, 0, s_wid, s_len);
  } else {
    println("Error: shape not defined.");
  }
  
  popMatrix(); // Restore the original drawing style settings and transformations

  // Draw the randomly rotated and sized rectangular occlusions, static in the scene
  if ((rand_occlusions) && (frameCount == 1)) {
    float sw = 1;
      for (int i = 0; i < num_occlusions; i++) {
        occlusionX[i] = (i%(num_occlusions/2))*width/4-width/4;//random(-width/2, width/2);
        if (i == num_occlusions/2) {sw = -1;}
        occlusionY[i] = sw*occlusionX[i];//random(-height/2, height/2);
        occlusionWidth[i] = width/14;//random(width/16, width/8);
        occlusionHeight[i] = height/2;//random(height/6, height/2);
        occ_colors[i] = color(random(255), random(255), random(255));
        if (i < num_occlusions/2) {occ_rot[i] = PI/2;} else {occ_rot[i] = 0;}//(i%(num_occlusions/2))*PI/3-PI/3;//random(PI/5, PI/3);
      }
  }

  if (rand_occlusions) {
    // int n = 30;
    // display_circles(n, n, width/(n*3)); // Display a grid of circles
    for (int i = 0; i < num_occlusions; i++) {
      pushMatrix();
      translate(width/2, height/2);
      rotate(occ_rot[i]);
      occ_colors[i] = color(random(255), random(255), random(255));
      fill(occ_colors[i]);
      strokeWeight(1);
      rect(occlusionX[i], occlusionY[i], occlusionWidth[i], occlusionHeight[i]);
      popMatrix();
    }
    fill(e_color); // Reset fill color
    strokeWeight(thickness);  // Reset stroke thickness
  }

  
  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }
  
  else if (save_frames == true) {
    if (train_mode) {saveFrame("frames/general_" + shape + "_" + mot_dir + "/###.png");}
    else if (test_mode) {saveFrame("frames/general_" + shape + "_" + mot_dir + "_test/###.png");}
    else {println("Error: train_mode and test_mode not defined."); exit();}
    if (frameCount == num_frames) {
      exit();
    }
  }

  else {
    if (frameCount == num_frames) {
      exit();
    }
  }
  
}

void display_circles (int n, int m, float s) {
  // a processing animation language function to display n x m circles of size s equally distributed over the screen height and width
  pushMatrix();
  float x, y;
  for (int i = 0; i < n + 1; i++) {
    for (int j = 0; j < m + 1; j++) {
      x = (i) * width / (n);
      y = (j) * height / (m);
      fill(color(random(255), random(255), random(255)));
      strokeWeight(s/5);
      ellipse(x, y, s, s);
    }
  }
  popMatrix();
  fill(e_color); // Reset fill color
  strokeWeight(thickness);  // Reset stroke thickness
}

// class Occ_array2D {
//     float[][] distances;
//     float maxDistance;
//     int spacer;
//     int w, h;

//     Occ_array2D (int width, int height) {
//         w = width;
//         h = height;
//         maxDistance = dist(w/2, h/2, w, h);
//         distances = new float[w][h];
//         for (int y = 0; y < h; y++) {
//             for (int x = 0; x < w; x++) {
//             float distance = dist(w/2, h/2, x, y);
//             distances[x][y] = distance/maxDistance * 255;
//             }
//         }
//         spacer = 10;
//         // noLoop();  // Run once and stop
//     }

//     void display() {
//         //   background(0);
//         // This embedded loop skips over values in the arrays based on
//         // the spacer variable, so there are more values in the array
//         // than are drawn here. Change the value of the spacer variable
//         // to change the density of the points
//         pushMatrix();
//         strokeWeight(6);
//         for (int y = 0; y < h; y += spacer) {
//             for (int x = 0; x < w; x += spacer) {
//             stroke(distances[x][y]);
//             point(x + spacer/2, y + spacer/2);
//             }
//         }
//         popMatrix();
//     }
// }

// class HLine { 
//   float ypos, speed; 
//   HLine (float y, float s) {  
//     ypos = y; 
//     speed = s; 
//   } 
//   void update() { 
//     ypos += speed; 
//     if (ypos > height) { 
//       ypos = 0; 
//     } 
//     line(0, ypos, width, ypos); 
//   } 
// } 



// import org.apache.commons.io.FileUtils;
// import java.io.File;

// String save_dir = ;
// if (train_mode == true) {String save_dir = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/general_cross_horizontal/frames/general_cross_horizontal/";}
// else if (test_mode == true) {String save_dir = "/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/general_cross_horizontal/frames/general_cross_horizontal_test/";}
// else {println("Error: train_mode and test_mode not defined."); exit();}

// void deleteDirectory(File dir) {
//   // Get all files in the directory
//   File[] files = dir.listFiles();
//   if (files != null) { // Some JVMs return null for empty dirs
//     for (File f : files) {
//       if (f.isDirectory()) {
//         deleteDirectory(f); // Recursively delete subdirectories
//       } else {
//         f.delete(); // Delete files
//       }
//     }
//   }
//   dir.delete(); // Delete the directory itself
// }