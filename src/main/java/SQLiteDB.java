import java.sql.*;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SQLiteDB {

    private static final Logger logger = LoggerFactory.getLogger(SQLiteDB.class);
    public static boolean dbExists = false;
    public static boolean dbEmpty = false;

    public static void createDB() {

        String dbUrl = "jdbc:sqlite:" + "./sqlite_databases/" + "memory_agent.db";

        try (Connection connection = DriverManager.getConnection(dbUrl);
             Statement statement = connection.createStatement()) {

            if (connection.isValid(30)) {

                System.out.println("Connection to database valid.");

            }

            // Check if the table exists
            String checkTableQuery = "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'";
            ResultSet tableResultSet = statement.executeQuery(checkTableQuery);

            if (!tableResultSet.next()) {
                // Table does not exist, create table
                String createTableQuery = "CREATE TABLE conversations (" +
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                        "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, " +
                        "prompt TEXT NOT NULL, " +
                        "response TEXT NOT NULL, " +
                        "prompt_embedding BLOB, " +
                        "response_embedding BLOB" +
                        ")";
                statement.executeUpdate(createTableQuery);
                System.out.println("Setting up memory database...done.");

                dbExists = true;
                dbEmpty = true;
            }


        } catch (SQLException e) {
            logger.error("Caught SQLException: {}", (Object) e.getStackTrace());
        }
    }

    public static void storeConversationWithEmbedding(String DB_URL,String prompt, String response, List<Double> prompt_embedding, List<Double> response_embedding) throws SQLException {
        String promptEmbeddingString = prompt_embedding.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(","));

        String responseEmbeddingString = response_embedding.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(","));

        try (Connection connection = DriverManager.getConnection(DB_URL);
             PreparedStatement pstmt = connection.prepareStatement(
                     "INSERT INTO conversations (prompt, response, prompt_embedding, response_embedding) VALUES (?, ?, ?, ?)")) {
            pstmt.setString(1, prompt);
            pstmt.setString(2, response);
            pstmt.setString(3, promptEmbeddingString);
            pstmt.setString(4, responseEmbeddingString);
            pstmt.executeUpdate();
            System.out.println("SYSTEM: Conversation saved to database.");

            dbEmpty = false;
        }
    }

}
