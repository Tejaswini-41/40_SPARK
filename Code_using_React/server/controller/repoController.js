import Repo from '../models/Repo.js';

// Add a new repository
export const addRepo = async (req, res) => {
  try {
    const { githubUrl, projectName, repoName } = req.body;

    // Validate input
    if (!githubUrl || !projectName || !repoName) {
      return res.status(400).json({ message: 'All fields are required' });
    }

    const newRepo = new Repo({
      githubUrl,
      projectName,
      repoName,
    });

    await newRepo.save();
    return res.status(201).json({ message: 'Repository added successfully', repo: newRepo });
  } catch (error) {
    console.error('Error adding repository:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

// Get all repositories
export const getRepos = async (req, res) => {
  try {
    const repos = await Repo.find();
    res.status(200).json(repos);
  } catch (error) {
    console.error('Error fetching repositories:', error);
    res.status(500).json({ message: 'Server error' });
  }
};
