package ab.dqn;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class logFile {
    public static String Read(String Path){
        String content = "";
        try {
            FileReader txtF = new FileReader(Path);
            BufferedReader readF = new BufferedReader(txtF);
            String line="";
            try {
                line = readF.readLine();
                while(line!=null){
                    content += line+"\n";
                    line = readF.readLine();
                }
                txtF.close();
                return content;
            } catch (IOException ex) {
                System.out.println("Erro: Failed to read File!");
                return "";
            }
        } catch (FileNotFoundException ex) {
            System.out.println("Erro: File not found!");
            return "";
        }
    }
    
    public static boolean Write(String Path,String Text){
        try {
            FileWriter txtW = new FileWriter(Path, true);
            PrintWriter writeF = new PrintWriter(txtW);
			writeF.println();
            writeF.print(Text);
            writeF.close();
            return true;
        }catch(IOException e){
            System.out.println(e.getMessage());
            return false;
        }
    }
}
