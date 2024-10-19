import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import connectDB from './config/db.js';  
import authRoutes from './routes/auth.js';
import repoRoutes from './routes/Repo.js';

dotenv.config();  
connectDB();

const app = express();

app.use(cors());
app.use(express.json());  

// Use the routes
app.use('/auth', authRoutes); 
app.use('/repo', repoRoutes);
app.post('/process', (req, res) => {
    res.send('Request received');
});

app.post('/ask', (req, res) => {
    res.send('Request received for question');
});
  


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

