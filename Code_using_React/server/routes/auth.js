import express from 'express';
import { loginUser, registerUser, getUserProfile } from '../controller/auth.js'; 
import { protect } from '../middleware/authMiddleware.js'; 

const router = express.Router();

router.post('/login', loginUser);
router.post('/register', registerUser);

// Route to get user profile
router.get('/profile', protect, getUserProfile); 

export default router;
