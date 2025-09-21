class BaseNet_CNN(object):
    def get_architecture(self):
        """
        Get the architecture of the network.

        Returns
        -------
        str
            Architecture of the network
        """
        return self.architecture
    
    def get_parameters(self):
        """
        Get the parameters of the network.

        Returns
        -------
        list
            List with the parameters of the network
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_num_params(self):
        """
        Get the number of parameters of the network.

        Returns
        -------
        int
            Number of parameters of the network
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)