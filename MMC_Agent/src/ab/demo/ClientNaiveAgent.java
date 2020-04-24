/*****************************************************************************xxx
 ** ANGRYBIRDS AI AGENT FRAMEWORK
 ** Copyright (c) 2014, XiaoYu (Gary) Ge, Stephen Gould, Jochen Renz
 **  Sahan Abeyasinghe,Jim Keys,  Andrew Wang, Peng Zhang
 ** MMC Agent: Murilo Mendel Costa
 ** All rights reserved.
**This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**To view a copy of this license, visit http://www.gnu.org/licenses/
 *****************************************************************************/
package ab.demo;

import java.awt.Color;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import ab.demo.HeartyLevelSelection;
import ab.demo.other.ClientActionRobot;
import ab.demo.other.ClientActionRobotJava;
import ab.planner.TrajectoryPlanner;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.Vision;
import ab.mlp.MLP;
import ab.mlp.logFile;

//Naive agent (server/client version)

public class ClientNaiveAgent implements Runnable {

	//Wrapper of the communicating messages
	private ClientActionRobotJava ar; //Encode Messages: agent to server / server to agent
	public byte currentLevel = -1; // Byte to represent the actual level
	public int failedCounter = 0; // Failed levels counter
	public int[] solved; // int list of solved levels
	TrajectoryPlanner tp;
	private int id = 22006; // Set Agent ID
	private boolean firstShot;
	private Point prevTarget; //Store last shot coordinate
	private HeartyLevelSelection levelSchemer;
	private Random randomGenerator;
	private MLP mlp;
	private logFile log; // Feed training Set File
	private int bufferedScore = 0;
	private int shotScore = 0;
	/**
	 * Constructor using the default IP
	 * */
	public ClientNaiveAgent() {
		// the default ip is the localhost
		ar = new ClientActionRobotJava("127.0.0.1");
		tp = new TrajectoryPlanner();
		randomGenerator = new Random();
		prevTarget = null;
		firstShot = true;
		mlp = new MLP();
	}
	/**
	 * Constructor with a specified IP
	 * */
	public ClientNaiveAgent(String ip) {
		ar = new ClientActionRobotJava(ip);
		tp = new TrajectoryPlanner();
		randomGenerator = new Random();
		prevTarget = null;
		firstShot = true;
		mlp	= new MLP();
	}
	public ClientNaiveAgent(String ip, int id)
	{
		ar = new ClientActionRobotJava(ip);
		tp = new TrajectoryPlanner();
		randomGenerator = new Random();
		prevTarget = null;
		firstShot = true;
		this.id = id;
		mlp = new MLP();
	}

    /* 
     * Run the Client (MMC Agent)
     */
	private void checkMyScore()
	{
		int[] scores = ar.checkMyScore();
		System.out.println(" My score: ");
		int level = 1;
		for(int i: scores)
		{
			System.out.println(" level " + level + "  " + i);
			if (i > 0)
				solved[level - 1] = 1;
			level ++;
		}
	}
	
	public void run() {	
		byte[] info = ar.configure(ClientActionRobot.intToByteArray(id)); // Register Team ID
		solved = new int[info[2]];
		
		//load the initial level (default 1)
		//Check my score
		checkMyScore();
		
		//Eagle Wing Level Selection Stategy
		levelSchemer = new HeartyLevelSelection(info, ar);
		System.out.println("Loading Level " + HeartyLevelSelection.currentLevel + "...");
		ar.loadLevel(HeartyLevelSelection.currentLevel); // Load the current level
		GameState state;
		while (true) {
			
			///Play the Game!!!!
			state = solve();

			if (state == GameState.WON) {
				
				///System.out.println(" loading the level " + (currentLevel + 1) );
				
				checkMyScore();
				System.out.println();
				
				//Eagle Wing Level Selection Stategy
				levelSchemer.updateStats(ar, true);
				ar.loadLevel(HeartyLevelSelection.currentLevel);
				
				//ar.loadLevel((byte)8);
				//display the global best scores
				int[] scores = ar.checkScore();
				System.out.println("Global best score: ");
				for (int i = 0; i < scores.length ; i ++)
				{
					System.out.print( " level " + (i+1) + ": " + scores[i]);
				}
				System.out.println();
				
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
				firstShot = true;
			} else 
				//If lost, then restart the level
				if (state == GameState.LOST) {
					
				levelSchemer.updateStats(ar, false);

				ar.loadLevel(HeartyLevelSelection.currentLevel);		
				} else if (state == GameState.LEVEL_SELECTION) {
					System.out.println("unexpected level selection page, go to the last current level : " + HeartyLevelSelection.currentLevel);
					ar.loadLevel(HeartyLevelSelection.currentLevel);
				} else if (state == GameState.MAIN_MENU) {
					System.out.println("unexpected main menu page, reload the level : " + HeartyLevelSelection.currentLevel);
					ar.loadLevel(HeartyLevelSelection.currentLevel);
				} else if (state == GameState.EPISODE_MENU) {
					System.out.println("unexpected episode menu page, reload the level: " + HeartyLevelSelection.currentLevel);
					ar.loadLevel(HeartyLevelSelection.currentLevel);
				}

		}
	}

	  /** 
	   * Solve a particular level by shooting birds directly to pigs
	   * @return GameState: the game state after shots.
     */
	public GameState solve() {
		
		String logContent = "";
		
		if(firstShot)
			bufferedScore = 0;
			
		// capture Image
		BufferedImage screenshot = ar.doScreenShot();
		
		// process image
		Vision vision = new Vision(screenshot);
		
		// find the slingshot
		Rectangle sling = vision.findSlingshotRealShape();

		System.out.println("in solve() before actionRobot.checkState()");
		GameState startState = ar.checkState();
		System.out.println("start state is playing? " + (startState == GameState.PLAYING));
		if (startState != GameState.PLAYING) {
			return startState;
		}
		
		//If the level is loaded (in PLAYING　state)but no slingshot detected, then the agent will request to fully zoom out.
		while (sling == null && ar.checkState() == GameState.PLAYING) {
			System.out.println("No slingshot detected. Please remove pop up or zoom out!");
		
			ar.fullyZoomIn();
			ar.fullyZoomOut();
			screenshot = ar.doScreenShot();
			vision = new Vision(screenshot);
			sling = vision.findSlingshotRealShape();
		}
		
		System.out.println("Getting screenshot to the class..");
		mlp.getInput(screenshot);		
		
		// get all the pigs
		List<ABObject> pigs = vision.findPigsRealShape();
		//System.out.println(pigs.size() + " Found Pigs");
		
		// find birds
		List<ABObject> birds = vision.findBirdsMBR();
		//System.out.println(birds.size() + " Found Birds");
		
		// get bird on sling
		ABObject bird_sling = new ABObject(sling, ar.getBirdTypeOnSling());
		
		// find blocks
		List<ABObject> blocks = vision.findBlocksRealShape();
		//System.out.println(blocks.size() + " Found Blocks");
		
		// find TNT
		List<ABObject> tnt = vision.findTNTs();		
		//System.out.println(tnt.size() + " Found TNTs");
		
		//Generating a list of all possible targets
		List<ABObject> targets = blocks;
		
		if(!pigs.isEmpty())
			for(int t=0; t < pigs.size(); t++)
				targets.add(pigs.get(t));
		
		if(!tnt.isEmpty())
			for(int t=0; t < tnt.size(); t++)
				targets.add(tnt.get(t));
		
		System.out.println("Pre-Prcessing Image..");
		mlp.preProcessImage(sling, birds, bird_sling, pigs, blocks, tnt);
		
		GameState state = ar.checkState();
		// if there is a sling, then play, otherwise skip.
		if (sling != null) {
			
			//If there are pigs, we call ANN to get a target. 
			if (!pigs.isEmpty()) {		
				Point releasePoint = null;
				
				//Pick up a target
				System.out.println("Making a Network Inference..");
				int[] tgt = new int[2];
				
				try{
					tgt = mlp.getTarget();
					System.out.println("From Python Target Coordinates: (" + tgt[0] + " , " + tgt[1] + ")");
				} catch (Exception e) {
					System.err.println("Failed to call getTarget Method!");
					e.printStackTrace();
				}
				
				// Using random target point selection
				Point _tpt = new Point(tgt[0], tgt[1]);
				//System.out.println("X coordinate" + _tpt.x);
				//System.out.println("Y coordinate" + _tpt.y);	
					
				//Adding target point log content.
				logContent += String.format("%03d %03d", _tpt.x, _tpt.y);
					
				prevTarget = new Point(_tpt.x, _tpt.y);

				// estimate the trajectory
				ArrayList<Point> pts = tp.estimateLaunchPoint(sling, _tpt);

				if (pts.size() == 1) {
					releasePoint = pts.get(0);
					
				} else if(pts.size() == 2) {
					
					int shotSelect = 0;
					if (shotSelect == 2){
						//releasePoint = pts.get(1); // Try High Shot
						releasePoint = pts.get(0); // Try Low Shot
					} else {
						releasePoint = pts.get(0); // Try low shot
					}
				}

				Point refPoint = tp.getReferencePoint(sling); // Reference Shot Point (focus_X,focus_y)

				// Get the release point from the trajectory prediction module
				int tapTime = 0;
				if (releasePoint != null) {
					double releaseAngle = tp.getReleaseAngle(sling,	releasePoint);
					
					System.out.println("Release Point: " + releasePoint);
					System.out.println("Release Angle: " + Math.toDegrees(releaseAngle));
					int tapInterval = 0;
					switch (ar.getBirdTypeOnSling()) 
					{
						case RedBird:
							tapInterval = 0; break;               // start of trajectory
						case YellowBird:
							//tapInterval = 65 + randomGenerator.nextInt(25);break; // 65-90% of the way
							tapInterval = 85;break; // 85% of the way
						case WhiteBird:
							tapInterval =  50 + randomGenerator.nextInt(20);break; // 50-70% of the way
						case BlackBird:
							tapInterval =  0;break; // 70-90% of the way
						case BlueBird:
							//tapInterval =  65 + randomGenerator.nextInt(20);break; // 65-85% of the way
							tapInterval = 95;break; // 85% of the way
						default:
							tapInterval =  85;
					}
						
					tapTime = tp.getTapTime(sling, releasePoint, _tpt, tapInterval);
						
				} else {
						System.err.println("No Release Point Found");
						return ar.checkState();
				}
				
				
				// check whether the slingshot is changed. the change of the slingshot indicates a change in the scale.
				ar.fullyZoomOut();
				screenshot = ar.doScreenShot();
				vision = new Vision(screenshot);
				Rectangle _sling = vision.findSlingshotRealShape();
				if(_sling != null) {
					
					double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
					if(scale_diff < 25) {
							
						int dx = (int) releasePoint.getX() - refPoint.x;
						int dy = (int) releasePoint.getY() - refPoint.y;
							
						if(dx < 0) {

							long timer = System.currentTimeMillis();
							ar.shoot(refPoint.x, refPoint.y, dx, dy, 0, tapTime, false); // shot in cartesian coordinates
							System.out.println("It takes " + (System.currentTimeMillis() - timer) + " ms to take a shot");
							while((System.currentTimeMillis() - timer) < 17000){
									//Waiting
							}
							state = ar.checkState();
							GameStateExtractor scoreExtractor = new GameStateExtractor();
								
							if ( state == GameState.PLAYING ) {

								screenshot = ar.doScreenShot();
								vision = new Vision(screenshot);
								shotScore = (scoreExtractor.getScoreInGame(screenshot) - bufferedScore); //Evaluate score from the actual shot									
								//System.out.println("P-Score for this shot are: " + shotScore);
								logContent += String.format(" %d", shotScore);
								List<Point> traj = vision.findTrajPoints();
								tp.adjustTrajectory(traj, sling, releasePoint);
								firstShot = false;
								bufferedScore += shotScore; // Update Score Buffer
								
								//System.out.println("Saving Log Content");
								//mlp.createLog(logContent); // Creating log Content for learning task
								
							} else if ( state == GameState.WON ) {
									
								screenshot = ar.doScreenShot();
								shotScore = scoreExtractor.getScoreEndGame(screenshot) - bufferedScore - ((birds.size() - 1) * 10000);
								logContent += String.format(" %d", shotScore);
								//System.out.println("W-Score for this shot are: " + shotScore);
								//System.out.println("Saving Log Content");
								//mlp.createLog(logContent); // Creating log Content for learning task
								
							}
						}
					}
					else
						System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
				} else
					System.out.println("no sling detected, can not execute the shot, will re-segement the image");	
			}
		}
		return state;
	}
	
	private double distance(Point p1, Point p2) {
		return Math.sqrt((double) ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y)* (p1.y - p2.y)));
	}
	
	public static void main(String args[]) {

		ClientNaiveAgent na;
		if(args.length > 0)
			na = new ClientNaiveAgent(args[0]);
		else
			na = new ClientNaiveAgent();
		na.run();
		
	}
}
