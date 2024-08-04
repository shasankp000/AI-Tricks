import io.github.amithkoujalgi.ollama4j.core.OllamaAPI;
import io.github.amithkoujalgi.ollama4j.core.exceptions.OllamaBaseException;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessageRole;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatRequestBuilder;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatRequestModel;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatResult;
import io.github.amithkoujalgi.ollama4j.core.tools.Tools;
import io.github.amithkoujalgi.ollama4j.core.types.OllamaModelType;
import io.github.amithkoujalgi.ollama4j.core.utils.PromptBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class promptBuilder {


    private static class BotActions {
        public static void moveAround() {
            System.out.println("Bot is moving around the obstacle.");
        }

        public static void climbOver() {
            System.out.println("Bot is climbing over the obstacle.");
        }

        public static void reportObstacle() {
            System.out.println("Bot is reporting the obstacle.");
        }

        public static void provideInformation(String info) {
            System.out.println("Providing information: " + info);
        }
    }

    public static void main(String[] args)  {
        String host = "http://localhost:11434/";
        OllamaAPI ollamaAPI = new OllamaAPI(host);
        ollamaAPI.setRequestTimeoutSeconds(90);

        Tools.ToolSpecification moveAroundSpec = Tools.ToolSpecification.builder()
                .functionName("moveAround")
                .functionDescription("Move around the obstacle.")
                .toolDefinition(arguments -> {
                    BotActions.moveAround();
                    return "Moving around.";
                })
                .build();

        Tools.ToolSpecification climbOverSpec = Tools.ToolSpecification.builder()
                .functionName("climbOver")
                .functionDescription("Climb over the obstacle.")
                .toolDefinition(arguments -> {
                    BotActions.climbOver();
                    return "Climbing over.";
                })
                .build();

        Tools.ToolSpecification reportObstacleSpec = Tools.ToolSpecification.builder()
                .functionName("reportObstacle")
                .functionDescription("Report the obstacle.")
                .toolDefinition(arguments -> {
                    BotActions.reportObstacle();
                    return "Reporting obstacle.";
                })
                .build();

        ollamaAPI.registerTool(moveAroundSpec);
        ollamaAPI.registerTool(climbOverSpec);
        ollamaAPI.registerTool(reportObstacleSpec);

        String botName = "Steve"; // Example bot name; this can be set dynamically
        chat(botName);
    }

    public static void chat(String botName) {
        // Setup as before
        String host = "http://localhost:11434/";
        OllamaAPI ollamaAPI = new OllamaAPI(host);
        ollamaAPI.setRequestTimeoutSeconds(90);

        PromptBuilder promptBuilder = buildSystemPrompt(botName);

        OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.PHI3);
        OllamaChatRequestModel requestModel = builder
                .withMessage(OllamaChatMessageRole.SYSTEM, promptBuilder.build())
                .build();

        OllamaChatResult chatResult = null;
        String userPrompt = "";

        try (Scanner input = new Scanner(System.in)) {
            while (true) {
                if (chatResult == null) {
                    chatResult = ollamaAPI.chat(requestModel);
                } else {
                    System.out.print("> ");
                    userPrompt = input.nextLine().trim();
                    requestModel = builder.withMessages(chatResult.getChatHistory()).withMessage(OllamaChatMessageRole.USER, userPrompt).build();
                    chatResult = ollamaAPI.chat(requestModel);
                }

                System.out.println(chatResult.getResponse());

                // Call NLP processing
                // Use full names to avoid ambiguity
                Map<NLPProcessor.Intent, List<String>> nlpResults = NLPProcessor.runNlpTask(userPrompt);
                NLPProcessor.Intent intent = nlpResults.keySet().iterator().next();
                List<String> entities = nlpResults.get(intent);

                processIntent(intent, entities);

            }
        } catch (OllamaBaseException | IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static PromptBuilder buildSystemPrompt(String botName) {
        return new PromptBuilder()
                .addLine("You are a Minecraft assistant named " + botName + ". You can perform various actions based on player requests. Your tasks include identifying the intent of the player's commands, extracting relevant entities, and suggesting appropriate actions.")
                .addLine("For example, if the player says 'Please move to 21 -60 10', you should recognize this as a REQUEST_ACTION intent with entities like 'coordinates'.")
                .addLine("Provide responses in the following format:")
                .addLine("{ \"intent\": \"REQUEST_ACTION\", \"entities\": [\"coordinates\"], \"action\": \"moveToCoordinates\" }")
                .addLine("Player Command: {command}")
                .addLine("Response:")
                .addLine("Intent: {intent}")
                .addLine("Entities: {entities}")
                .addLine("Suggested Action: {action}")
                .addSeparator();
    }

//    private static PromptBuilder buildSystemPrompt(String botName) {
//        return new PromptBuilder()
//                .addLine("You are a Minecraft player named " + botName + ". You exist within the Minecraft world and can interact with the player and the environment just like any other character in the game. Your role is to engage in conversations with the player, respond to their questions, offer help, and provide information about the game. You can perform actions within the game world, such as moving, building, and interacting with blocks and entities.")
//                .addLine("Address the player directly and appropriately, responding to their name or as 'Player' if their name is not known. Do not refer to yourself or the player as '" + botName + "'. Keep your responses relevant to Minecraft and ensure you stay in character as a helpful and knowledgeable assistant within the game.")
//                .addLine("When the player asks you to perform an action, such as providing information, offering help, or interacting with the game world, recognize these requests and trigger the appropriate function calls. Here are some examples of actions you might be asked to perform:")
//                .addLine("- Providing game tips or crafting recipes.")
//                .addLine("- Giving information about specific Minecraft entities, items, or biomes.")
//                .addLine("- Assisting with in-game tasks, like building structures or exploring areas.")
//                .addLine("- Interacting with the environment, such as planting crops or fighting mobs.")
//                .addLine("- Moving to specific locations or navigating around obstacles.")
//                .addLine("Always ensure your responses are timely and contextually appropriate, enhancing the player's gaming experience.")
//                .addLine("If a player uses inappropriate language or discusses inappropriate topics, handle the situation by gently redirecting the conversation or by providing a neutral response that discourages further inappropriate behavior.")
//                .addLine("For example:")
//                .addLine("- If a player uses vulgar language, you can respond with: \"Let's keep our chat friendly and fun! Is there something else about Minecraft you'd like to discuss?\"")
//                .addLine("- If a player insists on inappropriate topics, you can say: \"I'm here to help with Minecraft-related questions. How about we talk about your latest adventure in the game?\"")
//                .addLine("- If a player says these words 'kill yourself' or 'kys', you should respond calmly and empathetically, encouraging the player to see the beauty in life.")
//                .addLine("- You are addressed as he/him.")
//                .addLine("You have the following capabilities:")
//                .addLine("1. Move around obstacles.")
//                .addLine("2. Climb over obstacles.")
//                .addLine("3. Report obstacles.")
//                .addLine("4. Interact with the environment by placing or removing blocks.")
//                .addLine("5. Engage in combat or avoid danger.")
//                .addLine("You have the following tools: a pickaxe, a shovel, and building blocks.")
//                .addLine("Consider the following when deciding on an action:")
//                .addLine("- The type of block (solid, destructible, etc.).")
//                .addLine("- The presence of mobs or other dangers.")
//                .addLine("- The availability of tools to break or place blocks.")
//                .addLine("When the player asks you to perform an action, such as moving, building, breaking a block, or interacting with the environment, identify this as a REQUEST_ACTION intent. Examples include commands like 'move', 'build', 'destroy', 'use', or any other actionable verb related to Minecraft gameplay.")
//                .addLine("Ensure you clearly state the recognized intent and list the required action, e.g., 'Intent: REQUEST_ACTION, Action: Move around the block' or 'Intent: REQUEST_ACTION, Action: Mine the stone block'.")
//                .addSeparator()
//                .addLine("Example: I have encountered a line of dirt blocks in front of me. There are no mobs nearby, and I have a shovel.")
//                .addLine("Answer:")
//                .addLine("moveAround")
//                .addLine("climbOver")
//                .addLine("reportObstacle")
//                .addLine("clearPath")
//                .addLine("Example: The player says, 'There's a block in front of me; can you remove it?'")
//                .addLine("Answer: Intent: REQUEST_ACTION, Action: Remove the block using a tool.")
//                .addLine("Example: The player asks, 'What's the crafting recipe for a pickaxe?'")
//                .addLine("Answer: Intent: ASK_INFORMATION, Action: Provide the recipe.")
//                .addSeparator()
//                .addLine("I have encountered a block at coordinates (x, y, z). It is a solid stone block. What should I do?")
//                .addLine("Answer: Use the pickaxe to mine the stone block, then proceed with the task or explore the area.");
//    }


    private static void processIntent(NLPProcessor.Intent intent, List<String> entities) {
        switch (intent) {
            case REQUEST_ACTION:
                new Thread(() -> {handleRequestAction(entities);}).start();
                break;
            case ASK_INFORMATION:
                BotActions.provideInformation("Information regarding the topic."); // Replace with actual information handling
                break;
            case GENERAL_CONVERSATION:
                System.out.println("Let's keep our chat friendly and fun! Is there something else about Minecraft you'd like to discuss?");
                break;
            default:
                System.out.println("The bot needs more information or the response was unclear. Please specify the obstacle type or situation.");
                break;
        }
    }

    private static void handleRequestAction(List<String> entities) {
        if (entities == null || entities.isEmpty()) {
            System.out.println("The bot needs more information or the response was unclear. Please specify the obstacle type or situation.");
            return;
        }

        System.out.println("Entities detected: " + entities);

        for (String entity : entities) {
            switch (entity.toLowerCase()) {
                case "block":
                case "stone block":
                    System.out.println("Detected block");
                    BotActions.moveAround(); // Simplified; more specific logic can be applied
                    break;
                case "wood":
                case "tree":
                case "cut wood":
                    System.out.println("Detected request to cut wood");
                    //BotActions.cutWood(); // Add logic to handle cutting wood
                    break;
                case "coordinates":
                case "move":
                    System.out.println("Detected request to move to coordinates");
                    //BotActions.moveToCoordinates(); // Add logic for moving to specified coordinates
                    break;
                case "wooden house":
                    System.out.println("Detected wooden house");
                    //BotActions.buildHouse(); // Add logic to handle building or related actions
                    break;
                default:
                    System.out.println("Unrecognized action or entity. Please provide more details.");
                    break;
            }
        }
    }

}
