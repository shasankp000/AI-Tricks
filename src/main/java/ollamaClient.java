
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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Don't have the money for fine-tuning, so I will go with multi-shot learning here.

public class ollamaClient {

    private static final String DB_URL = "jdbc:sqlite:" + "./sqlite_databases/" + "memory_agent.db";
    private static final String host = "http://localhost:11434/";
    private static final Logger logger = LoggerFactory.getLogger(ollamaClient.class);

    private static final OllamaAPI ollamaAPI = new OllamaAPI(host);


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

                "\n The given data is just an EXAMPLE. DO NOT use any reference to it during conversations where context is not adequate or is missing. You must just learn the type of the context data accordingly to reply to the user's input if it seems relevant and has a high similarity score, particularly tending towards or greater than or even equalling 0.7" ;
    }


    public static void main(String[] args) throws SQLException, IOException, OllamaBaseException, InterruptedException {
        ollamaAPI.setRequestTimeoutSeconds(90);

        if (!SQLiteDB.dbExists) {

            RAGImplementation.init();

        }

        chat();
    }


    private static List<String> createQueries(String prompt) {
        String query_msg = "You are a first principle reasoning search query AI agent.\n" +
                "You must not portray any chatbot assistant like behaviour. Only execute the task which you are assigned to." +
                "Your list of search queries will be ran on an embedding database of all your conversations" +
                "you have ever had with the user. With first principles create a Python list of queries to 'search the embeddings database for any data that would be necessary" +
                " to have access to in order to correctly respond to the prompt. Your response must be a Python list with no syntax errors." +
                "Do not explain anything and do not ever generate anything but a perfect Python list of strings." +
                "\n Here are a few examples of an input json-like string you may receive." +
                "\n {'role': 'user', 'content': 'Write an email to my car insurance company and create a persuasive request for them to for them to lower my monthly rate" +
                "\n Or sometimes the input may just be a single sentence."+
                "\n For example, 'Based on everything that you know about me so far, tell me about myself.' or 'Hi there! What did we discuss about on last Tuesday?'" +


                "\n And here are a few examples of the type of queries you MUST generate in the form a Python-like list of strings." +
                "\n [\"What is the users name?\", \"What is the users current auto insurance provider?\", \"What is the monthly rate the user currently pays for auto insurance?]"+
                "\n [\"Discussion on Tuesday evening \", \" Topic of discussion\"]" +

                 "\n Remember that all queries should be IN A SINGLE Python list." +

                "\n Remember that the queries provided in this prompt are JUST EXAMPLES. You SHOULD NOT refer to these queries in the queries you generate UNLESS THE USER INPUT CORRESPONDS TO SUCH QUERIES." +

                "\n Only generate the query in the specified format. DO NOT say anything else or explain anything regarding the generated queries." ;
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

        String response;

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
            throw new RuntimeException(e);
        }

        return vectorDBQueries;
    }

    private static String recall(String userInput) throws SQLException {
        System.out.println("Recalling from memory.....");

        List<RAGImplementation.Conversation> conversationList = RAGImplementation.fetchConversations();
        Set<RAGImplementation.Conversation> relevantConversationSet = new HashSet<>();
        List<String> queryList = createQueries(userInput);

//        System.out.println("Query List: " + queryList.toString());

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

        StringBuilder finalRelevantConversationString = new StringBuilder();

        for (RAGImplementation.Conversation conversation : relevantConversationSet) {

                finalRelevantConversationString.append("ID: ").append(conversation.id).append("\n");
                finalRelevantConversationString.append(conversation.prompt).append("\n");
                finalRelevantConversationString.append(conversation.response).append("\n");
                finalRelevantConversationString.append(conversation.similarity).append("\n");


        }


        return finalRelevantConversationString.toString();

    }



    public static void chat() throws SQLException, IOException, OllamaBaseException, InterruptedException {

        Scanner scanner = new Scanner(System.in);

        OllamaChatRequestModel requestModel = null;
        OllamaChatResult chatResult = null;

        List<Double> promptEmbedding = new ArrayList<>();

        do {
            System.out.print("Enter your message: ");
            String userInput = scanner.nextLine().trim();

            if (SQLiteDB.dbEmpty) {



                if (requestModel == null) {

                    promptEmbedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, userInput);

                    OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
                    requestModel = builder
                            .withMessage(OllamaChatMessageRole.SYSTEM, generateSystemPrompt())
                            .withMessage(OllamaChatMessageRole.USER, "User input: " + userInput)
                            .build();


                }

                else {
                    promptEmbedding = ollamaAPI.generateEmbeddings(OllamaModelType.NOMIC_EMBED_TEXT, userInput);

                    OllamaChatRequestBuilder builder = OllamaChatRequestBuilder.getInstance(OllamaModelType.LLAMA2);
                    requestModel = builder
                            .withMessages(chatResult.getChatHistory())
                            .withMessage(OllamaChatMessageRole.USER, "User input: " + userInput)
                            .build();

                }

                chatResult = ollamaAPI.chat(requestModel);

            }

            else {

                String memoriesString = recall(userInput);

               // System.out.println(memoriesString);

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

