package ab.dqn;

import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.awt.Image;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import java.io.*;
import javax.imageio.ImageIO;

import ab.vision.ABObject;
import ab.vision.ABType;
import ab.dqn.logFile;


public class DQN {
	private BufferedImage ssImageR = null;
	private BufferedImage segImageR = null;
	private BufferedImage ssImage = null; // image obtained from screenshot process
	private BufferedImage segImage = null; // Cropped original Image
	private ABObject targetObj = null; // object chose as target
	private Point targetCoord = null; // target coordinates (x,y)
	private int imgCnt; // Contador para salvar imagens
	private logFile log; // Feed training Set File
	private Runtime rtT;
	private Runtime rtS;
	//pState // previous State (Criar uma classe para isso!!)
	//aState // actual state (Classe criada acima!!)
	//action // action to be executed (angle + target)
	//reward // reward from action (train network)

	
	public DQN() {
		this.imgCnt = 350;
		this.rtT = Runtime.getRuntime();
		this.rtS = Runtime.getRuntime();
	}
		
	//Receive the input image to be processed/trained
	public void getInput(BufferedImage im) {
		ssImage = copyImage(im);
		segImage = copyImage(im);
	}
	
	public static BufferedImage resize(BufferedImage img, int newW, int newH) {
		Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
		BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = dimg.createGraphics();
		g.drawImage(tmp, 0, 0, null);
		g.dispose();
		return dimg;
	}
	
	public void preProcessImage(Rectangle _sling, List<ABObject> _birds, ABObject _bird_sling, List<ABObject> _pigs, List<ABObject> _blocks, List<ABObject> _tnt) {
		
		fillBlack();
		placeSling(_sling, _bird_sling);
		placePigs(_pigs);
		placeBlocks(_blocks);
		placeTNTs(_tnt);
		placeBirds(_birds);
		
		segImage = segImage.getSubimage(119, 199, 720, 184);
		ssImage = ssImage.getSubimage(119, 199, 720, 184);
		
		ssImageR =  copyImage(ssImage);
		segImageR = copyImage(segImage);
		
		ssImageR = resize(ssImageR, 90, 23);
		segImageR = resize(segImageR, 90, 23);
	}
	
	private static BufferedImage copyImage(BufferedImage source){
		BufferedImage b = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());
		Graphics g = b.getGraphics();
		g.drawImage(source, 0, 0, null);
		g.dispose();
		return b;
	}
	
	//Place Sling in the new image to be used as the Deep Network Input
	private	void placeSling(Rectangle obj, ABObject brd) {
		
		for (int y = 0; y < obj.height; y++) {
			for (int x = 0; x< obj.width; x++) {
				
				//Filling bottom side of sling
				if (y > (obj.height / 2)) {
					segImage.setRGB(obj.x + x, obj.y + y, 6042391);
				} 
				
				// Filling top side of sling
				else {
					// Red Bird On Sling
					if (brd.type == ABType.RedBird) {
						segImage.setRGB(obj.x + x, obj.y + y, 16711680);
					}
					
					// Yellow Bird On Sling
					if (brd.type == ABType.YellowBird) {
						segImage.setRGB(obj.x + x, obj.y + y, 16776960);
					}
					
					// Blue Bird On Sling
					if (brd.type == ABType.BlueBird) {
						segImage.setRGB(obj.x + x, obj.y + y, 30975);
					}
					
					// Black Bird On Sling
					if (brd.type == ABType.BlackBird) {
						segImage.setRGB(obj.x + x, obj.y + y, 5921370);
					}
					
					// White Bird On Sling
					if (brd.type == ABType.WhiteBird) {
						segImage.setRGB(obj.x + x, obj.y + y, 16777215);
					}
				}
			}
		}	
	}
	
	//Place Pigs to new image to be used as the Deep Network Input
	private void placePigs(List<ABObject> obj) {
		
		if (!obj.isEmpty()) {
			for (int i=0; i < obj.size(); i++) {
				for (int y = 0; y < obj.get(i).height; y++) {
					for (int x = 0; x< obj.get(i).width; x++) {
						segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 6750130);
					}
				}
			}	
		}		
	}	
	
	//Place Birds to new image to be used as the Deep Network Input
	private void placeBirds(List<ABObject> obj) {
		
		if (!obj.isEmpty()) {
			for (int i=0; i < obj.size(); i++) {
				
				// Red Bird Case
				if (obj.get(i).type == ABType.RedBird) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 16711680);
						}
					}
				}
				
				// Yellow Bird Case
				if (obj.get(i).type == ABType.YellowBird) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 16776960);
						}
					}
				}
				
				// Blue Bird Case
				if (obj.get(i).type == ABType.BlueBird) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 30975);
						}
					}
				}
				
				// Black Bird Case
				if (obj.get(i).type == ABType.BlackBird) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 5921370);
						}
					}
				}
				
				// White Bird Case
				if (obj.get(i).type == ABType.WhiteBird) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 16777215);
						}
					}
				}
				
			}	
		}
	}
	
	//Place Birds to new image to be used as the Deep Network Input
	private void placeBlocks(List<ABObject> obj) {
		
		if (!obj.isEmpty()) {
			for (int i=0; i < obj.size(); i++) {
				
				// Stone Case
				if (obj.get(i).type == ABType.Stone) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 11053224);
						}
					}
				}
				
				// Wood Case
				if (obj.get(i).type == ABType.Wood) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 14255897);
						}
					}
				}
				
				// Ice Case
				if (obj.get(i).type == ABType.Ice) {
					for (int y = 0; y < obj.get(i).height; y++) {
						for (int x = 0; x< obj.get(i).width; x++) {
							segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 49151);
						}
					}
				}
				
			}	
		}
	}
	
	//Place TNTs to new image to be used as the Deep Network Input
	private void placeTNTs(List<ABObject> obj) {
		
		if (!obj.isEmpty()) {
			for (int i=0; i < obj.size(); i++) {
				for (int y = 0; y < obj.get(i).height; y++) {
					for (int x = 0; x< obj.get(i).width; x++) {
						segImage.setRGB(obj.get(i).x + x, obj.get(i).y + y, 16744192);
					}
				}
			}	
		}
	}
	
	//Set all pixels in black
	private void fillBlack() {
		
		for (int y = 0; y < segImage.getHeight(); y++) {
			for (int x = 0; x < segImage.getWidth(); x++) {	
				segImage.setRGB(x, y, 0);
			}
		}	
	}
	
	public int[] getTarget() throws IOException {
		
		// Saving screenShot to test directory
		String imgFilepath = String.format(System.getProperty("user.dir") + "/dataSet/prediction/Target/predictionImg.png");
		String imgFilename = String.format("predictionImg.png");
		String response = "";
		
		try {
			ImageIO.write(segImageR, "png", new File(imgFilepath));
		} catch (Exception e) {
			System.err.println("failed to save image " + imgFilepath);
			e.printStackTrace();
		}
		System.out.println("");
		//Shot command path
		//String pyTargetCommand = "python ./src/ab/dqn/targetDQN1.py";
		String pyTargetCommand = "python ./targetDQN.py";
		
		// Setting script caller command parameters
		String cmd = pyTargetCommand + " " + imgFilename;
		System.out.println("command called: " + cmd);
		
		//Create runtime to execute external command
		Process target_pr = rtT.exec(cmd);
		
		//Retrieve output from script
		BufferedReader target_bfr = new BufferedReader(new InputStreamReader(target_pr.getInputStream()));
		String line = "";
		while((line = target_bfr.readLine()) != null) {
			System.out.println(line);
			response = line;
		}

		target_pr.destroy();
		
		System.out.println("Python response is: " + response);
		
		// Set call response
		int pyZone = Integer.parseInt(response);
		int[] targetC = new int[2];
		int[] targetR = new int[2];
		System.out.println("Actual Zone = " + pyZone);
		int xZone = (pyZone-1) - ((pyZone-1) / 14) * 14;
		int yZone = (pyZone-1) / 14;
		System.out.println("xZone = " + xZone + " yZone = " + yZone);
		
		Random ran = new Random();
		int ranX = ran.nextInt(8);
		int ranY = ran.nextInt(8);
		int xPixel = (((xZone * 3) + 1 + 40) * 8) + 4 + 119;
		int yPixel = (((yZone * 3) + 1) * 8) + 4 + 199;
		
		targetC[0] = xPixel;
		targetC[1] = yPixel;
		
		System.out.println("Target Coordinates Real = (" + targetC[0] +  "," + targetC[1] + ")");
		
		targetR[0] = ((targetC[0] - 119) / 8) - 40;
		targetR[1] = ((targetC[1] - 199) / 8);
		
		targetR[0] = targetR[0] / 3;
		targetR[1] = targetR[1] / 3;
		
		System.out.println("Recovered Target Coordinates Resized = (" + (targetR[0]) +  "," + targetR[1] + ")");
		
		int ZONE = targetR[0] + (targetR[1]*14) + 1;
		System.out.println("Recovered Zone = " + ZONE);
		return targetC;
	}
	
	public int[] getTargetRand() throws IOException {
		
		// Saving screenShot to test directory
		String imgFilepath = String.format(System.getProperty("user.dir") + "/dataSet/prediction/Target/predictionImg.png");
		String imgFilename = String.format("predictionImg.png");
		String response = "";
		
		try {
			ImageIO.write(segImageR, "png", new File(imgFilepath));
		} catch (Exception e) {
			System.err.println("failed to save image " + imgFilepath);
			e.printStackTrace();
		}
		System.out.println("");
		//Shot command path
		String pyTargetCommand = "python ./src/ab/dqn/targetDQN1.py";
		
		// Setting script caller command parameters
		String cmd = pyTargetCommand + " " + imgFilename;
		System.out.println("command called: " + cmd);
		
		//Create runtime to execute external command
		Process target_pr = rtT.exec(cmd);
		
		//Retrieve output from script
		BufferedReader target_bfr = new BufferedReader(new InputStreamReader(target_pr.getInputStream()));
		String line = "";
		while((line = target_bfr.readLine()) != null) {
			System.out.println(line);
			response = line;
		}

		target_pr.destroy();
		
		System.out.println("Python response is: " + response);
		
		// Set call response
		int pyZone = Integer.parseInt(response);
		int[] targetC = new int[2];
		int[] targetR = new int[2];
		System.out.println("Actual Zone = " + pyZone);
		int xZone = (pyZone-1) - ((pyZone-1) / 14) * 14;
		int yZone = (pyZone-1) / 14;
		System.out.println("xZone = " + xZone + " yZone = " + yZone);
		
		Random ran = new Random();
		int ranX = ran.nextInt(8);
		int ranY = ran.nextInt(8);
		int xPixel = (((xZone * 3) + ran.nextInt(3) + 40) * 8) + ranX + 119;
		int yPixel = (((yZone * 3) + ran.nextInt(3)) * 8) + ranY + 199;
		
		targetC[0] = xPixel;
		targetC[1] = yPixel;
		
		System.out.println("Target Coordinates Real = (" + targetC[0] +  "," + targetC[1] + ")");
		
		targetR[0] = ((targetC[0] - 119) / 8) - 40;
		targetR[1] = ((targetC[1] - 199) / 8);
		
		targetR[0] = targetR[0] / 3;
		targetR[1] = targetR[1] / 3;
		
		System.out.println("Recovered Target Coordinates Resized = (" + (targetR[0]) +  "," + targetR[1] + ")");
		
		int ZONE = targetR[0] + (targetR[1]*14) + 1;
		System.out.println("Recovered Zone = " + ZONE);
		return targetC;
	}
	
	public void createLog(String logInfo) {
		
		//String imgFilename1 = String.format(System.getProperty("user.dir") + "/dataSet/trainSet/original/img%06d.png", imgCnt);
		//String imgFilename2 = String.format(System.getProperty("user.dir") + "/dataSet/trainSet/originalResized/img%06d.png", imgCnt);
		String imgFilename3 = String.format(System.getProperty("user.dir") + "/dataSet/trainSet/segmented/img%06d.png", imgCnt);
		String imgFilename4 = String.format(System.getProperty("user.dir") + "/dataSet/trainSet/segmentedResized/img%06d.png", imgCnt);
		String imgName = String.format("img%06d.png", imgCnt);
		String logPath = String.format(System.getProperty("user.dir") + "/dataSet/trainSet/dqnShot.txt");
		String logContent = imgName + " " + logInfo;
		
		try {
			//ImageIO.write(ssImage, "png", new File(imgFilename1));
			//ImageIO.write(ssImageR, "png", new File(imgFilename2));
			ImageIO.write(segImage, "png", new File(imgFilename3));
			ImageIO.write(segImageR, "png", new File(imgFilename4));
			imgCnt++;
			log.Write(logPath, logContent);
		} catch (Exception e) {
			System.err.println("failed to save image " + imgFilename3);
			e.printStackTrace();
		}
	}
}