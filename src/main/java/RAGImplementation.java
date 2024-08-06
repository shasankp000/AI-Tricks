import java.sql.*;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RAGImplementation {
    private static final Logger logger = LoggerFactory.getLogger(RAGImplementation.class);

    public static class Conversation {
        // DS for the SQL return type.
        int id;
        String prompt;
        String response;
        List<Double> promptEmbedding;
        List<Double> responseEmbedding;
        double similarity;

        Conversation(int id, String prompt, String response, List<Double> promptEmbedding, List<Double> responseEmbedding) {
            this.id = id;
            this.prompt = prompt;
            this.response = response;
            this.promptEmbedding = promptEmbedding;
            this.responseEmbedding = responseEmbedding;
        }
    }

    public static void init() {
        // create the database and table if not already exists.

        SQLiteDB.createDB();

    }

    private static List<Double> parseEmbedding(String embeddingString) {
        String[] parts = embeddingString.split(",");
        List<Double> embedding = new ArrayList<>();
        for (String part : parts) {
            embedding.add(Double.parseDouble(part));
        }
        return embedding;
    }

    public static void printEmbedding(String label, List<Double> embedding) {
        System.out.println(label + ": length = " + embedding.size());
        for (Double value : embedding) {
            if (value.isNaN()) {
                System.out.println("Found NaN value in " + label);
            }
        }
    }

    public static List<Conversation> findRelevantConversations(List<Double> queryEmbedding, List<Conversation> conversations, int topN) {
        for (Conversation conv : conversations) {
            double promptSimilarity = calculateCosineSimilarity(queryEmbedding, conv.promptEmbedding);
            double responseSimilarity = calculateCosineSimilarity(queryEmbedding, conv.responseEmbedding);
            conv.similarity = (promptSimilarity + responseSimilarity) / 2; // Average similarity
        }
        conversations.sort((c1, c2) -> Double.compare(c2.similarity, c1.similarity)); // Sort in descending order
        return conversations.subList(0, Math.min(topN, conversations.size()));
    }

    // Cosine similarity where if the angle between two vectors overlap, they are similar (angle = 0)
    // If angle is 90 then the vectors are dissimilar.

    // cos_sim(x,y) = [(x.y) / [|x| . |y|]]

    public static double calculateCosineSimilarity(List<Double> vec1, List<Double> vec2) {
        if (vec1.size() != vec2.size()) {
            throw new IllegalArgumentException("Vectors must be of the same length");
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        for (int i = 0; i < vec1.size(); i++) {
            dotProduct += vec1.get(i) * vec2.get(i);
            norm1 += Math.pow(vec1.get(i), 2);
            norm2 += Math.pow(vec2.get(i), 2);
        }
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    public static List<Conversation> fetchConversations() throws SQLException {
        // fetch all conversations from the database.

        String DB_URL = "jdbc:sqlite:" + "./sqlite_databases/" + "memory_agent.db";

        List<Conversation> conversations = new ArrayList<>();
        try (Connection connection = DriverManager.getConnection(DB_URL);
             Statement statement = connection.createStatement();
             ResultSet resultSet = statement.executeQuery("SELECT id, prompt, response, prompt_embedding, response_embedding FROM conversations")) {

            while (resultSet.next()) {
                int id = resultSet.getInt("id");
                String prompt = resultSet.getString("prompt");
                String response = resultSet.getString("response");
                String promptEmbeddingString = resultSet.getString("prompt_embedding");
                String responseEmbeddingString = resultSet.getString("response_embedding");
                List<Double> promptEmbedding = parseEmbedding(promptEmbeddingString);
                List<Double> responseEmbedding = parseEmbedding(responseEmbeddingString);

                conversations.add(new Conversation(id, prompt, response, promptEmbedding, responseEmbedding));
            }
        }
        catch (Exception e) {
            logger.error("Caught exception: {}", (Object) e.getStackTrace());
        }

        return conversations;
    }


}
