
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.amithkoujalgi.ollama4j.core.OllamaAPI;
import io.github.amithkoujalgi.ollama4j.core.exceptions.OllamaBaseException;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessageRole;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatRequestBuilder;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatRequestModel;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatResult;

import io.github.amithkoujalgi.ollama4j.core.types.OllamaModelType;


import java.io.IOException;

import java.sql.SQLException;

import java.util.*;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Don't have the money for fine-tuning, so I will go with multi-shot learning here.

public class ollamaClient {

    private static final String DB_URL = "jdbc:sqlite:" + "./sqlite_databases/" + "memory_agent.db";
    private static final String host = "http://localhost:11434/";
    private static final Logger logger = LoggerFactory.getLogger(ollamaClient.class);

    private static final OllamaAPI ollamaAPI = new OllamaAPI(host);
    private static final DateTimeFormatter formatter1 = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
    private static final DateTimeFormatter formatter2 = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public static String generateSystemPrompt() {
        return "You are an AI assistant that is supposed to have memory of every conversation you have ever had with this user. " +
                "On every prompt from the user, the system will check for any relevant messages you have had with the user. " +
                "If any embedded previous conversations are attached, use them for context to respond to the user, " +
                "if the context is relevant and useful to responding. If the recalled conversations are irrelevant or if there are no previous conversations at all, " +
                "disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. " +
                "Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant." +

                "\n And here is an EXAMPLE of the type of context which you will receive: " +
                "\n \"ID: 1\"" +
                "\n \"Prompt: Hi there! My name is John Doe. How is your day today?\"" +
                "\n \"Response: Hi John! I'm doing well, thank you for asking! It's great to be chatting with you again. Can you tell me more about what brings you here today? Are you working on a specific project or just looking for some general information?" +
                "\n \"Similarity: 0.40820406249846664\"" +
                "\n \"ID: 2 \"" +
                "\n \"Prompt: How can i convert the speak function in my llama3 python voice assistant to use pyttsx3? \""+
                "\n \"Response: To convert the speak function in your LLaMA3 Python voice assistant to use pyttsx3, you'll need to follow these steps: \n 1. Install pyttsx3 if you haven't already: \n pip install pyttsx3 \n 2. Import pyttsx3 in your Python script. \n 3. Initialize the pyttsx3 engine. \n 4. Create a speak function using pyttsx3."+

                "\n The given data is just an EXAMPLE. DO NOT use any reference to it during conversations where context is not adequate or is missing. You must just learn the type of the context data accordingly to reply to the user's input." +

                "\n HOWEVER if you receive this context message instead of the above examples: \n" +

                "No relevant data found from database. Analyze current user input and answer accordingly." +

                "\n Then this means that based on the current user input, no relevant data was found on the database. So you must analyse the current user input and answer accordingly. Be it providing answer to a new question or solving a math problem." +

               "\n You will also be given the current date and time at which the user asks the current question to you. Use this information for sorting through the context data for the most recent conversation or if the user asks you a question about a topic from a specific date.";

    }


    public static void main(String[] args) throws SQLException, IOException, OllamaBaseException, InterruptedException {
        ollamaAPI.setRequestTimeoutSeconds(90);

        if (!SQLiteDB.dbExists) {

            RAGImplementation.init();

        }

        chat();
    }

    private static boolean isRecentTimestamp(String timestamp, String currentDateTime) {
        LocalDateTime conversationTime = LocalDateTime.parse(timestamp, formatter2);
        LocalDateTime currentTime = LocalDateTime.parse(currentDateTime, formatter1);

        // Within the last hour
        return conversationTime.isAfter(currentTime.minusHours(1));
    }

    private static double findMaxSimilarity(Set<RAGImplementation.Conversation> conversationSet) {
        double maxSimilarity = Double.MIN_VALUE;

        for (RAGImplementation.Conversation conversation : conversationSet) {
            if (conversation.similarity > maxSimilarity) {
                maxSimilarity = conversation.similarity;
            }
        }

        return maxSimilarity;
    }

    private static boolean isHighSimilarity(double similarityScore, double maxSimilarity) {
        return similarityScore == maxSimilarity;
    }

    private static String getCurrentDateandTime() {

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
        LocalDateTime now = LocalDateTime.now();

        return dtf.format(now);

    }


    private static boolean classify_conversations(String DateTime, String prompt, String retrieved_context) {

        boolean isRelevant = false;

        String sys_prompt = """
                You are an conversation classification AI agent. Your inputs will be a prompt, the current date and time when the prompt was asked and a chunk of text that has the following parameters: \n
                1. ID: This is just an ID pertaining to it's order in the database. If the id's number is small then it refers to early conversations.
                2. Timestamp: This is the timestamp at which each conversation is recorded. This is useful for analysis if the current user prompt asks something related to "most recent conversation" or a conversation on a specific date.
                3. Prompt: The question asked/statement made by the user.
                4. Response: The response made by the language model in use at that time of recording the conversation.
                5. Similarity: The similarity score which is obtained after a vector similarity calculation. The closer it is to 1, the better the chances of the conversation being similar to the current user input.
              \n
               \n
               You will not respond as an AI assistant. You only respond "yes" or "no". \n
               \n
                Determine whether the context contains data that directly is related to the search query. \n
               \n
                If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly 'related respond "no". Do not respond "yes" unless the content is highly relevant to the search query. \n
               \n
                Here's an example of the prompts you may receive: \n
               \n
               Example 1: Based on everything that you know about me so far, tell me about myself. \n
               Example 2: What is the derivative of x^y with respect to y? \n
               Example 3: What did we discuss last Tuesday? \n
               Example 4: What is the weather in Bengaluru right now? \n
               \n
               And here's the type of context data you will receive: \n
              \n
               "\\n \\"ID: 1\\"" +
               "\\n \\"Prompt: Hi there! My name is John Doe. How is your day today?\\"" +
               "\\n \\"Response: Hi John! I'm doing well, thank you for asking! It's great to be chatting with you again. Can you tell me more about what brings you here today? Are you working on a specific project or just looking for some general information?" +
               "\\n \\"Similarity: 0.40820406249846664\\"" +
               "\\n \\"ID: 2 \\"" +
               "\\n \\"Prompt: How can i convert the speak function in my llama3 python voice assistant to use pyttsx3? \\""+
               "\\n \\"Response: To convert the speak function in your LLaMA3 Python voice assistant to use pyttsx3, you'll need to follow these steps: \\n 1. Install pyttsx3 if you haven't already: \\n pip install pyttsx3 \\n 2. Import pyttsx3 in your Python script. \\n 3. Initialize the pyttsx3 engine. \\n 4. Create a speak function using pyttsx3."+
              \n
           \n
              Note how in this example the prompts and context are completely unrelated. That doesn't mean this will always be the case. You must analyse both the prompts and context data properly and return only a single word answer. Yes or No, irrespective of case. \n
             \n
              If you receive no such data, then it means there is no "probable relevant data" to classify. In that case simply say No, irrespective of case.
             \n
      \s
           \s""";

        String userEnd = "This is the user prompt: " + prompt;
        String contextData = "This is the context data from the database: " + "\n" + retrieved_context;

        OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
        OllamaChatRequestModel requestModel2 = builder
                .withMessage(OllamaChatMessageRole.SYSTEM, sys_prompt)
                .withMessage(OllamaChatMessageRole.USER, userEnd)
                .withMessage(OllamaChatMessageRole.USER, contextData)
                .withMessage(OllamaChatMessageRole.USER, "Current date and time: " + DateTime)
                .build();

        String response;


        try {
            OllamaChatResult chatResult1 = ollamaAPI.chat(requestModel2);

            response = chatResult1.getResponse();

            System.out.println("Conversation classifier: " + "\n" + response);

            if (response.equalsIgnoreCase("yes") || response.startsWith("Yes") || response.startsWith("yes") || response.contains("Yes") || response.contains("yes")) {

                isRelevant = true;

            }


        } catch (OllamaBaseException | IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }


        return isRelevant;
    }

    private static List<String> createQueries(String prompt) {
        String query_msg = "You are a first-principles reasoning search query AI agent. Your task is to generate a list of queries to search an embeddings database for relevant data. The output must be a valid Python list of strings in JSON format. Here are a few examples of input strings you may receive:\n" +
                           "[\"Based on everything that you know about me so far, tell me about myself.\", \"What did we discuss on last Tuesday?\", \"Tell me about implicit differentiation in multivariable calculus\"]\n" +
                           "And here are examples of the format your output must follow:\n" +
                           "[\"What is the user's name?\", \"What did the user ask on yyyy/mm/dd hh:mm:ss?\"]\n" +
                           "The output must be a single Python list of strings in JSON format, with no additional explanations or syntax errors. The queries must be directly related to the user's input. If you receive a date or timestamp, format it as 'yyyy/mm/dd hh:mm:ss'. Do not generate anything other than the list.";

        // + "\n Especially explaining your response like \" Here are some Python list queries to search the embeddings database for necessary information to correctly respond to the prompt: \". Such responses should not be in, before or after the final output of the list of queries";


        List<Map<String, String>> queryConvo = new ArrayList<>();
        Map<String, String> queryMap1 = new HashMap<>();

        List<String> vectorDBQueries = new ArrayList<>();

        queryMap1.put("role", "system");
        queryMap1.put("content", query_msg);

        queryConvo.add(queryMap1);

        OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
        OllamaChatRequestModel requestModel1 = builder
                .withMessage(OllamaChatMessageRole.SYSTEM, queryConvo.toString())
                .withMessage(OllamaChatMessageRole.USER, prompt)
                .build();

        String response = "";

        try {
            OllamaChatResult chatResult1 = ollamaAPI.chat(requestModel1);

            response = chatResult1.getResponse();


            int startIndex = response.indexOf('[');
            int endIndex = response.lastIndexOf(']') + 1;

            if (startIndex != -1 && endIndex != -1) {
                String listString = response.substring(startIndex, endIndex);

                ObjectMapper objectMapper = new ObjectMapper();
                vectorDBQueries = objectMapper.readValue(listString, new TypeReference<List<String>>() {
                });

                return vectorDBQueries;
            }

        } catch (OllamaBaseException | IOException | InterruptedException e) {
            logger.error("Caught exception: {} ", (Object) e.getStackTrace());
            System.out.println(response);
            throw new RuntimeException(e);
        }

        return vectorDBQueries;
    }

    private static String recall(String DateTime, String userInput) throws SQLException {
        System.out.println("Recalling from memory.....");

        List<RAGImplementation.Conversation> conversationList = RAGImplementation.fetchConversations();
        Set<RAGImplementation.Conversation> relevantConversationSet = new HashSet<>();
        List<String> queryList = createQueries(userInput);

        System.out.println("Query List: " + queryList.toString());

        List<Double> embedding;

        for (String query : queryList) {
            try {
                embedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, query);
                List<RAGImplementation.Conversation> relevantConversations = RAGImplementation.findRelevantConversations(embedding, conversationList, 2);
                relevantConversationSet.addAll(relevantConversations);
            } catch (IOException | InterruptedException | OllamaBaseException e) {
                logger.error("Caught new exception: {}", (Object) e.getStackTrace());
                throw new RuntimeException(e);
            }
        }

        String finalRelevantConversationString;

        StringBuilder relevantConversationString = new StringBuilder();

        double maxSimilarity = findMaxSimilarity(relevantConversationSet);

        for (RAGImplementation.Conversation conversation : relevantConversationSet) {
            boolean isRecent = isRecentTimestamp(conversation.timestamp, DateTime);
            boolean isHighSimilarity = isHighSimilarity(conversation.similarity, maxSimilarity);

            // Consider the conversation relevant if it's recent and has the highest similarity,
            // or if it's not recent but has the highest similarity
            if ((isRecent && isHighSimilarity) || (!isRecent && isHighSimilarity)) {
                relevantConversationString.append("ID: ").append(conversation.id).append("\n");
                relevantConversationString.append("Timestamp: ").append(conversation.timestamp).append("\n");
                relevantConversationString.append(conversation.prompt).append("\n");
                relevantConversationString.append(conversation.response).append("\n");
                relevantConversationString.append(conversation.similarity).append("\n");
            }
        }


        if (classify_conversations(DateTime, userInput,relevantConversationString.toString())) {

            finalRelevantConversationString = relevantConversationString.toString();
        }

        else {

            finalRelevantConversationString = "No relevant data found from database. Analyze current user input and answer accordingly.";

        }

        return finalRelevantConversationString;

    }



    public static void chat() throws SQLException, IOException, OllamaBaseException, InterruptedException {

        Scanner scanner = new Scanner(System.in);

        OllamaChatRequestModel requestModel = null;
        OllamaChatResult chatResult = null;

        List<Double> promptEmbedding = new ArrayList<>();

        String dateTime;

        System.out.println("System initialized at: " + getCurrentDateandTime());

        do {
            System.out.print("Enter your message: ");
            String userInput = scanner.nextLine().trim();

            dateTime = getCurrentDateandTime();

            if (SQLiteDB.dbEmpty) {

                if (requestModel == null) {

                    promptEmbedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, userInput);

                    OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
                    requestModel = builder
                            .withMessage(OllamaChatMessageRole.SYSTEM, generateSystemPrompt())
                            .withMessage(OllamaChatMessageRole.USER, "User input: " + userInput)
                            .withMessage(OllamaChatMessageRole.USER, "Current date and time: " + dateTime)
                            .build();


                }

                else {
                    promptEmbedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, userInput);

                    OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
                    requestModel = builder
                            .withMessages(chatResult.getChatHistory())
                            .withMessage(OllamaChatMessageRole.USER, "User input: " + userInput)
                            .withMessage(OllamaChatMessageRole.USER, "Current date and time: " + dateTime)
                            .build();

                }

                chatResult = ollamaAPI.chat(requestModel);

            }

            else {

                String memoriesString = recall(dateTime,userInput);

                System.out.println(memoriesString);

                promptEmbedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, userInput);

                if (chatResult == null) {

                    OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
                    requestModel = builder
                            .withMessage(OllamaChatMessageRole.SYSTEM, generateSystemPrompt())
                            .withMessage(OllamaChatMessageRole.USER, "User input: " + userInput)
                            .withMessage(OllamaChatMessageRole.USER, "Memories: " + memoriesString)
                            .build();

                    chatResult = ollamaAPI.chat(requestModel);

                }
                else {

                    OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
                    requestModel = builder
                            .withMessages(chatResult.getChatHistory())
                            .withMessage(OllamaChatMessageRole.USER, "User input: " + userInput)
                            .withMessage(OllamaChatMessageRole.USER, "Memories: " + memoriesString)
                            .build();

                    chatResult = ollamaAPI.chat(requestModel);


                }


            }

            String response;

            try {

                response = chatResult.getResponse();

                System.out.println(response);

                List<Double> responseEmbedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, userInput);

                SQLiteDB.storeConversationWithEmbedding(DB_URL, userInput, response, promptEmbedding, responseEmbedding);


            } catch (OllamaBaseException | IOException | InterruptedException e) {
                logger.error("Caught exception: {} ", (Object) e.getStackTrace());
                throw new RuntimeException(e);
            }


            // Exit condition check
            System.out.println("Type 'exit' to close the chat or press return to continue.");
        } while (!scanner.nextLine().trim().equalsIgnoreCase("exit"));


    }


}

