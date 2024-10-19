// server/routes/repo.js
import express from 'express';
import { addRepo, getRepos } from '../controller/repoController.js';

const router = express.Router();

// Route to add a new repository
router.post('/', addRepo);

// Route to get all repositories
router.get('/all', getRepos);

export default router;
