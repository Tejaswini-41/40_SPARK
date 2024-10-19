import React, { useState, useEffect } from 'react';
import axios from 'axios';


const Sidebar = ({ onRepoSelect }) => {
  const [repoList, setRepoList] = useState([]);
  const [formData, setFormData] = useState({ githubUrl: '', projectName: '', repoName: '' });

  useEffect(() => {
    axios.get('/api/repos')
      .then(response => setRepoList(response.data))
      .catch(error => console.error('Error fetching repositories:', error));
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleAddRepo = () => {
    axios.post('/api/repos', formData)
      .then(response => {
        setRepoList([...repoList, response.data]);
        setFormData({ githubUrl: '', projectName: '', repoName: '' });
      })
      .catch(error => console.error('Error adding repository:', error));
  };

  return (
    <div className="w-64 bg-gray-800 text-white p-4">
      <div className="mb-4">
        <h2 className="text-lg font-bold">User Profile</h2>
        {/* Display user profile details here */}
      </div>
      <div className="mb-4">
        <h2 className="text-lg font-bold">Add Repository</h2>
        <input 
          type="text" 
          name="githubUrl" 
          value={formData.githubUrl} 
          onChange={handleInputChange} 
          placeholder="GitHub URL" 
          className="w-full mb-2 p-2"
        />
        <input 
          type="text" 
          name="projectName" 
          value={formData.projectName} 
          onChange={handleInputChange} 
          placeholder="Project Name" 
          className="w-full mb-2 p-2"
        />
        <input 
          type="text" 
          name="repoName" 
          value={formData.repoName} 
          onChange={handleInputChange} 
          placeholder="Repository Name" 
          className="w-full mb-2 p-2"
        />
        <button onClick={handleAddRepo} className="bg-blue-500 p-2 w-full">
          Add Repository
        </button>
      </div>
      <div>
        <h2 className="text-lg font-bold">Repositories</h2>
        {repoList.map((repo) => (
          <div 
            key={repo._id} 
            className="bg-gray-700 p-2 my-2 cursor-pointer" 
            onClick={() => onRepoSelect(repo)}
          >
            {repo.projectName}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
