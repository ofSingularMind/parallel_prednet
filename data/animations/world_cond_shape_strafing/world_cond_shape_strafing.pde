ArrayList<Shape> shapes;
World the_world;
int num_shapes = 1;
float x, y;
int lastAddedFrame = 0;
int addInterval = 900; // Time in milliseconds to add new shape
int speedDivider = 8; // Bigger is slower
int world_state_divider = 12;
int num_occlusions = 6;
int frame_rate; //<>//
int num_frames;
PImage[] images;
float[] occlusionX = new float[num_occlusions];
float[] occlusionY = new float[num_occlusions];
float[] occlusionWidth = new float[num_occlusions];
float[] occlusionHeight = new float[num_occlusions];
color[] occ_colors = new color[num_occlusions];
float[] occ_rot = new float[num_occlusions];
boolean rand_occlusions = true;
int ws = 50;

String rand_background = "pixels"; // can be "pixels", "whole", "white"

boolean save_gif = false; // only set save_gif or save_frames to true, not both, or both to false
boolean save_frames = true;
boolean second_stage = true; // switches to white background and grey occlusions to sharpen up predictions

boolean train_mode = false; // just flip this one to switch between train and test modes
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
  shapes = new ArrayList<Shape>();
  rectMode(CENTER);
  stroke(0);
  // Set the frame rate and the number of frames to be saved per mode
  if (save_gif && save_frames) {println("Error: save_gif and save_frames cannot both be true."); exit();}
  if (save_gif) {num_frames = 150; frame_rate = 200;}
  else if (save_frames && train_mode) {num_frames = 30000; frame_rate = 5000;} // deleteDirectory(new File(save_dir));}
  else if (save_frames && test_mode) {num_frames = 1200; frame_rate = 1000;} // deleteDirectory(new File(save_dir));}
  else {num_frames = 1000; frame_rate = 10;}
  images = new PImage[num_frames];

  frameRate(frame_rate);
  if (save_gif == true) {
    if (train_mode && !second_stage) {gifExport = new GifMaker(this, "world_cond_shape_train_1st_stage.gif");}
    else if (test_mode && !second_stage) {gifExport = new GifMaker(this, "world_cond_shape_test_1st_stage.gif");}
    else if (train_mode && second_stage) {gifExport = new GifMaker(this, "world_cond_shape_train_2nd_Stage.gif");}
    else if (test_mode && second_stage) {gifExport = new GifMaker(this, "world_cond_shape_test_2nd_Stage.gif");}
    else {println("Error: train_mode and test_mode not defined."); exit();}
    gifExport.setRepeat(0); // make it an "endless" animation
    // gifExport.setTransparent(255); // make white the transparent color -- match browser bg color
    gifExport.setDelay(1000/frame_rate);  //12fps in ms
  }
  if ((rand_background == "pixels") && (second_stage == false)) {
    for (int i = 0; i < num_frames; i++) {
      images[i] = loadImage("/home/evalexii/Documents/Thesis/code/parallel_prednet/data/animations/backgrounds/"+nf(ws)+"x"+nf(ws)+"/" + nf(i%10000, 3) + ".png");
    }
  }

  // Create world
  the_world = new World();

}

void draw() {
  // Set the background to random pixels or white:
  if ((rand_background == "pixels") && (second_stage == false)) {
    image(images[frameCount-1], 0, 0, width, height);
  } else if ((rand_background == "whole") && (second_stage == false)) {
    background(color(random(100, 200), random(100, 200), random(100, 200)));
  } else if ((rand_background == "white") && (second_stage == false)) {
    background(255);
  } else if (second_stage == true) {
    background(255); // white anyways
  }

  // Manage the world
  the_world.update();

  // Manage the shapes
  if (frameCount - lastAddedFrame > random(1) && shapes.size() < num_shapes) {
    color c = color(255, 0, 0);
    float rot = random(PI);
    float stroke = random(0.1, 2);
    if (flip) { //(random(1) < 0.5) {
      float[] len_wid = get_len_wid(1.5);
      float len = len_wid[0];
      float wid = len_wid[1];
      // crosses go down (x,y)
      for (int i = 0; i < int(random(5)); i++) {
        x = random(width*0.1, width*0.9);
        y = (len/2); // + height/world_state_divider;
        // y = random(len + height/world_state_divider, height*0.5);
      }
      shapes.add(new Cross(x, y, len, wid, rot, c, stroke, the_world));
      flip = false;
    } else {
      float[] len_wid = get_len_wid(1);
      float len = len_wid[0];
      float wid = len_wid[1];
      // ellipses go right (x,y)
      for (int i = 0; i < int(random(5)); i++) {
        x = (len/2); // + width/world_state_divider;
        // x = random(len + width/world_state_divider, width*0.5);
        y = random(height*0.1, height*0.9);
      }
      shapes.add(new Ellipse(x, y, len, wid, rot, c, stroke, the_world));
      flip = true;
    }
    lastAddedFrame = frameCount;
  }

  for (int i = 0; i < shapes.size(); i++) {
    Shape shape = shapes.get(i);
    shape.update();
    shape.display();
  }

  for (int i = shapes.size() - 1; i >= 0; i--) {
    Shape shape = shapes.get(i);
    if (!shape.isActive()) {
      shapes.remove(i);
    }
  }

  // Display world state on top
  the_world.display();

  if (save_gif == true) {
    gifExport.addFrame();
  
    if (frameCount == num_frames) {
      gifExport.finish();
      exit();
    }
  }
  
  else if (save_frames == true) {
    if (!second_stage) {
      if (train_mode) {saveFrame("frames/world_cond_shape_1st_stage/###.png");}
      else if (test_mode) {saveFrame("frames/world_cond_shape_test_1st_stage/###.png");}
    }
    else if (second_stage) {
      if (train_mode) {saveFrame("frames/world_cond_shape_2nd_stage/###.png");}
      else if (test_mode) {saveFrame("frames/world_cond_shape_test_2nd_stage/###.png");}
    }
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

class World {
    boolean world_state;

    World(){
        this.world_state = true;
    }

    boolean get_world_state(){
        return this.world_state;
    }

    void update(){
        if (random(1) < 0.33) {
            this.world_state = !this.world_state;
        }
    }

    void display(){
        // rectMode(CORNERS);
        if (this.world_state) {
            fill(0, 255, 0);
            // background(0, 255, 0);
        } else {
            fill(255, 0, 0);
            // background(255, 0, 0);
        }
        strokeWeight(1);
        // Left column
        rect(1*width/5, height/2, width/7, height/7);
        
        // Middle column
        rect(width/2, 1*height/5, width/7, height/7);
        rect(width/2, height/2, width/7, height/7);
        rect(width/2, 4*height/5, width/7, height/7);
        
        // Right column
        rect(4*width/5, height/2, width/7, height/7);
    }
}

abstract class Shape {
  float x, y, len, wid, rotation, stroke;
  color c;
  boolean active = true;
  boolean state = true;
  World the_world;
  
  abstract void update();
  abstract void display();
  boolean isActive() {
    return active;
  }
}

class Cross extends Shape {
  Cross(float x, float y, float len, float wid, float rotation, color c, float stroke, World the_world) {
    this.x = x;
    this.y = y;
    this.len = len;
    this.wid = wid;
    this.rotation = rotation;
    this.c = c;
    this.stroke = stroke;
    this.state = true;
    this.the_world = the_world;
  }
  
  void update() {
    if (state) {
        y += (height/speedDivider);
        if (y > height + len/2) active = false;
    }
    state = the_world.get_world_state();
  }
  
  void display() {
    if (state) {fill(0);}
    else {fill(0);}
    strokeWeight(stroke);
    pushMatrix();
    translate(x, y);
    rotate(rotation);
    rect(0, 0, wid, len); // Vertical line
    rect(0, 0, len, wid); // Horizontal line
    popMatrix();
  }
}

class Ellipse extends Shape {
  Ellipse(float x, float y, float len, float wid, float rotation, color c, float stroke, World the_world) {
    this.x = x;
    this.y = y;
    this.len = len;
    this.wid = wid;
    this.rotation = rotation;
    this.c = c;
    this.stroke = stroke;
    this.state = true;
    this.the_world = the_world;
  }
  
  void update() {
    if (state) {
        x += (width/speedDivider);
        if (x > width + len/2) active = false;
    }
    state = the_world.get_world_state();
  }
  
  void display() {
    if (state) {fill(0);}
    else {fill(0);}
    strokeWeight(stroke);
    pushMatrix();
    translate(x, y);
    rotate(rotation);
    ellipse(0, 0, wid, len);
    popMatrix();
  }
}

public float[] get_len_wid(float multiplier) {
    float d = random(40, 70);
    if (train_mode || test_mode) {
      while (d >= 50 && d <= 60) {d = random(40, 70);}
    } else if (test_mode == true) {
      while (!(d > 52 && d < 58) && !(d > 73 && d < 90)) {d = random(53, 90);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    float s_wid_divisor = random(2, 4);
    if (train_mode || test_mode) {
      while (s_wid_divisor >= 3 && s_wid_divisor <= 3.75) {s_wid_divisor = random(2, 4);}
    } else if (test_mode == true) {
      while (!(s_wid_divisor > 3.25 && s_wid_divisor < 3.5) && !(s_wid_divisor > 3.5 && s_wid_divisor < 5)) {s_wid_divisor = random(3.25, 5);}
    } else {
      println("Error: train_mode and test_mode not defined."); exit();
    }
    // if ((shape == "cross") || (shape == "rectangle")) {s_wid_divisor = s_wid_divisor * 1.5;} 
    // for debug with window_size != 50
    if (width != 50.0) {d = d * (width / 50.0);}
    float s_len = d/1.5;
    float s_wid = s_len / (s_wid_divisor*multiplier);
    return new float[] {s_len, s_wid};
}


// class Occlusions {
//     int num_occlusions;
//     float thickness;
//     float rotation;
//     float x, y;

//     Occlusions(){
//         this.num_occlusions = 5;
//         this.thickness = height/this.num_occlusions;
//         this.rotation = PI/4;
//         this.x = 0;
//         this.y = 0;
//     }      
    
//     void display(){
//         fill(127, 127, 127);
//         strokeWeight(stroke);
//         pushMatrix();
//         translate(x, y);
//         rotate(rotation);
//         rect(0, 0, wid, len); // Vertical line
//         popMatrix();

//         // display a set of alternating rectangles and gaps
//         for (int i = 0; i < num_occlusions; i++) {

//             occlusionX[i] = random(width);
//             occlusionY[i] = random(height);
//             occlusionWidth[i] = random(10, 50);
//             occlusionHeight[i] = random(10, 50);
//             occ_colors[i] = color(random(255), random(255), random(255));
//             occ_rot[i] = random(PI);
//             fill(occ_colors[i]);
//             rect(occlusionX[i], occlusionY[i], occlusionWidth[i], occlusionHeight[i]);
//         }
//     }
// }
