// Kudos to this guy, Matt Williams, https://www.youtube.com/watch?v=IdPdwQdM9lA, for opening my eyes on function calling.


import com.google.gson.*;
import com.google.gson.stream.JsonReader;
import io.github.amithkoujalgi.ollama4j.core.OllamaAPI;
import io.github.amithkoujalgi.ollama4j.core.exceptions.OllamaBaseException;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessageRole;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatRequestBuilder;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatRequestModel;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatResult;
import io.github.amithkoujalgi.ollama4j.core.types.OllamaModelType;

import java.io.IOException;
import java.io.StringReader;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class FunctionCallerV2 {

    public static class ExecutionRecord {
        String command;
        String parameters;
        String result;

        public ExecutionRecord(String command, String parameters, String result) {
            this.command = command;
            this.parameters = parameters;
            this.result = result;
        }
    }



    public static class DummyTools {

        public static void goTo(int x, int y, int z) {
            System.out.println("Going to coordinates " + x + " " + y + " " + z);
        }

        public static void check_block(String direction) {
            switch (direction) {
                case "front" -> System.out.println("Checking for block in front");
                case "behind" -> {
                    System.out.println("Rotating..");
                    System.out.println("Checking for block behind");
                }
                case "left" -> {
                    System.out.println("Rotating..");
                    System.out.println("Checking for block in left");
                }
                case "right" -> {
                    System.out.println("Rotating..");
                    System.out.println("Checking for block in right");
                }
            }
        }

    }

    public static String toolBuilder() {

        List<Map<String, Object>> functions = new ArrayList<>();

        functions.add(buildFunction("goTo", "Move to coordinates", List.of(
                buildParameter("x", "The x-axis coordinate", true),
                buildParameter("y", "The y-axis coordinate", true),
                buildParameter("z", "The z-axis coordinate", true)
        )));

        functions.add(buildFunction("checkBlock", "Check block in direction", List.of(
                buildParameter("direction", "The direction to check (front, behind, left, right)", true)
        )));

        functions.add(buildFunction("mineBlock", "Mine block in direction", List.of(
                buildParameter("direction", "The direction to mine (front, behind, left, right)", true)
        )));

        functions.add(buildFunction("attackEntity", "Attack a specific entity", List.of(
                buildParameter("entityName", "The name of the entity to attack", true)
        )));

        functions.add(buildFunction("useItem", "Use a specific item", List.of(
                buildParameter("itemName", "The name of the item to use", true)
        )));

        Map<String, Object> toolMap = new HashMap<>();
        toolMap.put("functions", functions);

        return new Gson().toJson(toolMap);
    }

    private static Map<String, Object> buildFunction(String name, String description, List<Map<String, Object>> parameters) {
        Map<String, Object> function = new HashMap<>();
        function.put("name", name);
        function.put("description", description);
        function.put("parameters", parameters);
        return function;
    }

    private static Map<String, Object> buildParameter(String name, String description, boolean required) {
        Map<String, Object> parameter = new HashMap<>();
        parameter.put("name", name);
        parameter.put("description", description);
        parameter.put("required", required);
        return parameter;
    }

    // This code right here is pure EUREKA moment.

    public static String buildPrompt(String toolString) {
        return "You are a helpful assistant that takes a question and finds " +
                "the most appropriate tool or tools to execute, along with the " +
                "parameters required to run the tool. Respond as JSON using the " +
                "following schema: {" +
                "\"functionName\": \"function name\", " +
                "\"parameters\": [" +
                        "{\"" +
                            "parameterName\": \"name of parameter\", " +
                            "\"parameterValue\": \"value of parameter\"" +
                        "}" +
                    "]" +
                "}" + "Return the json with proper indentation so that there are no parsing errors. DO NOT modify the json field names. It is absolutely imperative that the field names are NOT MODIFIED." +
                "The tools are: " + toolString +
                "While returning the json output, do not say anything else. By anything else, I mean any other word at all.";
    }

    public static String formatFeedback(String command, String parameters, String result) {
        return String.format("Command: %s, Parameters: %s, Result: %s", command, parameters, result);
    }

    public static String buildFeedbackPrompt(List<ExecutionRecord> executionHistory) {
        StringBuilder feedback = new StringBuilder("Here is the history of the commands executed so far and their results:\n");
        for (ExecutionRecord record : executionHistory) {
            feedback.append(formatFeedback(record.command, record.parameters, record.result)).append("\n");
        }
        feedback.append("Use this information to improve your responses.");
        return feedback.toString();
    }


    public static void main(String[] args) {
        String host = "http://localhost:11434/";
        OllamaAPI ollamaAPI = new OllamaAPI(host);
        ollamaAPI.setRequestTimeoutSeconds(90);

        OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2); // LLAMA2 is surprisingly much less error prone compared to phi3.

        String prompt = buildPrompt(toolBuilder());

        OllamaChatResult chatResult = null;
        Scanner scanner = new Scanner(System.in);
        Gson gson = new Gson();
        String response = "";

        List<ExecutionRecord> executionHistory;
        String feedbackPrompt = "";

        while (true) {
            System.out.print("Enter message: ");
            String userInput = scanner.nextLine().trim();

            try {
                if (chatResult != null) {
                    // Send feedback about the previous action result

                    OllamaChatRequestModel requestModel = builder
                            .withMessages(chatResult.getChatHistory())
                            .withMessage(OllamaChatMessageRole.SYSTEM, "Previous action result: " + feedbackPrompt)
                            .withMessage(OllamaChatMessageRole.USER, userInput)
                            .build();

                    chatResult = ollamaAPI.chat(requestModel);
                    response = chatResult.getResponse();
                    System.out.println(response);

                } else {
                    // Initial prompt and function call
                    OllamaChatRequestModel requestModel = builder
                            .withMessage(OllamaChatMessageRole.SYSTEM, prompt)
                            .withMessage(OllamaChatMessageRole.USER, userInput)
                            .build();

                    chatResult = ollamaAPI.chat(requestModel);
                    response = chatResult.getResponse();
                    System.out.println(response);
                }


                executionHistory = executeFunction(response);
                feedbackPrompt = buildFeedbackPrompt(executionHistory);
                // Call the function and capture the result


            } catch (OllamaBaseException | IOException | InterruptedException | JsonSyntaxException e) {
                e.printStackTrace();
            }

            System.out.println("Enter 'exit' to quit.");
            String exitInput = scanner.nextLine().trim();
            if (exitInput.equalsIgnoreCase("exit")) {
                break;
            }
        }

    }

    public static String cleanJsonString(String jsonString) {
        // Remove ```json and ``` markers
        jsonString = jsonString.replaceAll("```json", "").replaceAll("```", "").trim();

        // Remove non-printable characters
        jsonString = jsonString.replaceAll("[^\\x20-\\x7E]", "").replaceAll("\\\\n", "").replaceAll("\\s+", " ");

        // Attempt to correct common JSON structure errors
        jsonString = correctParameterNames(jsonString);

        // Ensure proper JSON format
        jsonString = jsonString.replaceAll("\\s*:\\s*", ":").replaceAll("\\s*,\\s*", ",");
        jsonString = jsonString.replaceAll("\\}\\s*\\]", "}]");

        // If the JSON still seems malformed, attempt to manually correct it
        if (!isValidJson(jsonString)) {
            jsonString = attemptManualCorrection(jsonString);
        }

        return jsonString;
    }

    public static String correctParameterNames(String jsonString) {
        // Fix parameter names in a malformed JSON string
        jsonString = jsonString.replaceAll("\"name\":", "\"parameterName\":");
        jsonString = jsonString.replaceAll("\"value\":", "\"parameterValue\":");

        // Fix other potential issues
        Pattern pattern = Pattern.compile("\"parameterName\\d+\":\"([a-zA-Z]+)\",\\s*\"parameterValue\":\"([^\"]+)\"");
        StringBuffer sb = getStringBuffer(jsonString, pattern);

        return sb.toString();
    }

    private static StringBuffer getStringBuffer(String jsonString, Pattern pattern) {
        Matcher matcher = pattern.matcher(jsonString);
        StringBuffer sb = new StringBuffer();

        int counter = 0;
        while (matcher.find()) {
            matcher.appendReplacement(sb, "\"parameterName\":\"" + matcher.group(1) + "\",\"parameterValue\":\"" + matcher.group(2) + "\"");
            counter++;
        }
        matcher.appendTail(sb);

        // Ensure the parameter array is correctly closed
        if (counter > 0 && !jsonString.endsWith("}]")) {
            sb.append("}]");
        }
        return sb;
    }

    public static boolean isValidJson(String jsonString) {
        try {
            JsonReader reader = new JsonReader(new StringReader(jsonString));
            reader.setLenient(true);
            JsonParser.parseReader(reader).getAsJsonObject();
            return true;
        } catch (JsonSyntaxException | IllegalStateException e) {
            return false;
        }
    }

    public static String attemptManualCorrection(String jsonString) {
        // Attempt to manually correct known issues with the JSON string
        jsonString = jsonString.replaceAll("\"parameterName\\d+\":", "\"parameterName\":");
        jsonString = jsonString.replaceAll("\"parameterValue([a-zA-Z]+)\":", "\"parameterValue\":");

        // Fix trailing commas and other common mistakes
        jsonString = jsonString.replaceAll(",\\s*}", "}");
        jsonString = jsonString.replaceAll(",\\s*]", "]");

        return jsonString;
    }


    public static List<ExecutionRecord> executeFunction(String response) {

        List<ExecutionRecord> executionHistory = new ArrayList<>();

        try {
            String cleanedResponse = cleanJsonString(response);
            System.out.println("Cleaned JSON Response: " + cleanedResponse); // Log the cleaned JSON response for debugging
            JsonReader reader = new JsonReader(new StringReader(cleanedResponse));
            reader.setLenient(true);
            JsonObject jsonObject = JsonParser.parseReader(reader).getAsJsonObject();
            String functionName = jsonObject.get("functionName").getAsString();
            JsonArray parameters = jsonObject.get("parameters").getAsJsonArray();

            StringBuilder params = new StringBuilder();
            Map<String, String> parameterMap = new HashMap<>();

            for (JsonElement parameter : parameters) {
                JsonObject paramObj = parameter.getAsJsonObject();
                String paramName = paramObj.get("parameterName").getAsString();
                String paramValue = paramObj.get("parameterValue").getAsString();
                params.append(paramName).append("=").append(paramValue).append(", ");

                parameterMap.put(paramName, paramValue);

            }

            // Simulate function execution
            String result = "Executed " + functionName + " with parameters " + params.toString();
            String actionOutput = callFunction(functionName, parameterMap);
            System.out.println(result + "\n " + "Output: " + actionOutput);

            // Store execution record
            executionHistory.add(new ExecutionRecord(functionName, params.toString(), result));
        } catch (JsonSyntaxException | NullPointerException e) {
            System.err.println("Error processing JSON response: " + e.getMessage());
        }


        return executionHistory;
    }


    public static String callFunction(String functionName, Map<String, String> paramMap) {
        switch (functionName) {
            case "goTo":
                int x = Integer.parseInt(paramMap.get("x"));
                int y = Integer.parseInt(paramMap.get("y"));
                int z = Integer.parseInt(paramMap.get("z"));
                DummyTools.goTo(x, y, z);
                return "Moved to coordinates (" + x + ", " + y + ", " + z + ").";

            case "checkBlock":
                String direction = paramMap.get("direction");
                DummyTools.check_block(direction);
                return "Checked for block in " + direction + " direction.";

            default:
                return "Unknown function: " + functionName;
        }
    }

}
