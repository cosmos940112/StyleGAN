package edu.southwestern.tasks.mario;

import java.util.List;

import ch.idsia.ai.agents.Agent;
import ch.idsia.mario.engine.sprites.Mario;
import ch.idsia.tools.CmdLineOptions;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.EvaluationOptions;
import ch.idsia.tools.Evaluator;
import edu.southwestern.MMNEAT.MMNEAT;
import edu.southwestern.evolution.EvolutionaryHistory;
import edu.southwestern.evolution.genotypes.Genotype;
import edu.southwestern.evolution.genotypes.TWEANNGenotype;
import edu.southwestern.networks.Network;
import edu.southwestern.networks.NetworkTask;
import edu.southwestern.networks.TWEANN;
//import edu.southwestern.networks.hyperneat.HyperNEATTask;
//import edu.southwestern.networks.hyperneat.Substrate;
//import edu.southwestern.networks.hyperneat.SubstrateConnectivity;
import edu.southwestern.parameters.CommonConstants;
import edu.southwestern.parameters.Parameters;
import edu.southwestern.tasks.NoisyLonerTask;
import edu.southwestern.util.datastructures.Pair;
import edu.southwestern.util.random.RandomNumbers;

public class MarioTask<T extends Network> extends NoisyLonerTask<T> implements NetworkTask {

	private EvaluationOptions options;
	public static final int MARIO_OUTPUTS = 5; //need to find a way to make sure this isn't hardcoded

	public MarioTask(){
    	options = new CmdLineOptions(new String[0]);
        options.setLevelDifficulty(Parameters.parameters.integerParameter("marioLevelDifficulty"));
        options.setMaxFPS(!CommonConstants.watch); // Run fast when not watching
        options.setVisualization(CommonConstants.watch);
        options.setTimeLimit(Parameters.parameters.integerParameter("marioTimeLimit"));
        MMNEAT.registerFitnessFunction("Progress");
        
        if(Parameters.parameters.booleanParameter("moMario")){
        	 MMNEAT.registerFitnessFunction("Time");
        }
	}
	
	/**
	 * @returns number of objectives that the controller is being evaluated on,
	 * 			always at least 1: progress, but sometimes 2: progress and time
	 */
	@Override
	public int numObjectives() {
		if(Parameters.parameters.booleanParameter("moMario")){
			return 2;
		} else {
			return 1;
		}
	}

	@Override
	public double getTimeStamp() {
		// Not sure we can use this? -Gab
		return 0;
	}

	@Override
	public String[] sensorLabels() {
		int xStart = Parameters.parameters.integerParameter("marioInputStartX");
		int yStart = Parameters.parameters.integerParameter("marioInputStartY");
		int width = Parameters.parameters.integerParameter("marioInputWidth");
		int height = Parameters.parameters.integerParameter("marioInputHeight");
		int xEnd = height + xStart;
		int yEnd = width + yStart;
		int worldBuffer = 0;
		int enemiesBuffer = (width * height);
		String[] labels = new String[((width * height) * 2) + 1];
		for(int x = xStart; x < xEnd; x++){
			for(int y = yStart; y < yEnd; y++){
				labels[worldBuffer++] = "Object at (" + x + ", " + y + ")";
				labels[enemiesBuffer++] = "Enemy at (" + x + ", " + y + ")";
			}
		}
		labels[enemiesBuffer++] = "Bias";		
		return labels;
	}

	@Override
	public String[] outputLabels() {
		return new String[]{"Left", "Right", "Down", "Jump", "Speed"};
		//Note: These may not be correct, as there are only 5/6 -Gab
	}

	/**
	 * @param individual
	 * 			individual to be tested
	 * 	@param num
	 * 			evaluation number
	 * @return results of the evaluation, distance traveled by individual (+ time if multi-objective)
	 */
	@Override
	public Pair<double[], double[]> oneEval(Genotype<T> individual, int num) {
		Pair<double[], double[]> evalResults;
		double distanceTravelled = 0;
		double timeSpent = 0;
		options.setAgent(new NNMarioAgent<T>(individual));
		if(Parameters.parameters.booleanParameter("deterministic"))
			options.setLevelRandSeed(num); //generates from the same seeds for every individual.
		else
			options.setLevelRandSeed(RandomNumbers.randomGenerator.nextInt(Integer.MAX_VALUE));
		Evaluator evaluator = new Evaluator(options);
		List<EvaluationInfo> results = evaluator.evaluate();
		
		for (EvaluationInfo result : results) {
			distanceTravelled += result.computeDistancePassed();
			timeSpent = result.timeSpentOnLevel;
			if(result.marioStatus == Mario.STATUS_WIN){
				timeSpent = result.totalTimeGiven;
			} 
		}
		distanceTravelled /= results.size();
		if(Parameters.parameters.booleanParameter("moMario")){
			evalResults = new Pair<double[], double[]>(new double[] { distanceTravelled, timeSpent }, new double[0]);
		} else {
			evalResults = new Pair<double[], double[]>(new double[] { distanceTravelled }, new double[0]);			
		}
		return evalResults;
	}
	
	/**
	 * Setter for the task options 
         * @param options settings about evaluation
	 */
    public void setOptions(EvaluationOptions options) {
        this.options = options;
    }

    /**
     * Getter for the task options
     * @return EvaluationOptions options 
     */
    public EvaluationOptions getOptions() {
        return options;
    }

    /**
     * Simple test of MarioTask
     * @param args 
     */
    public static void main(String[] args){
    	Parameters.initializeParameterCollections(new String[]{"io:false", "netio:false", 
    			"task:edu.southwestern.tasks.mario.MarioTask", "marioInputStartX:-3", "marioInputStartY:-2", 
    			"marioInputWidth:12", "marioInputHeight:5", "showMarioInputs:false", "moMario:true", "marioJumpTimeout:20", "marioStuckTimeout:100"});
    	MMNEAT.loadClasses();
    	EvolutionaryHistory.initArchetype(0);
    	TWEANNGenotype tg = new TWEANNGenotype(10,5,0);
    	MarioTask<TWEANN> mt = new MarioTask<TWEANN>();
    	Agent controller = new NNMarioAgent<TWEANN>(tg);
    	
    	EvaluationOptions options = new CmdLineOptions(new String[0]);
        options.setAgent(controller);
        
        options.setMaxFPS(false);
        options.setVisualization(true);
        options.setNumberOfTrials(1);
        options.setMatlabFileName("");
        options.setLevelRandSeed((int) (Math.random () * Integer.MAX_VALUE));
        options.setLevelDifficulty(3);
        mt.setOptions(options);
    	
    	mt.oneEval(tg, 0);
    	
    }

}
