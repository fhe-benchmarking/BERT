
import torch
import tqdm
import os

# This is the training function. Train, show the loss and accuracy for every epoch.
def train_model_function(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)  # Move the model to the specified device (GPU or CPU)
    
    # Lists to store loss and accuracy for each epoch
    epoch_losses = []
    epoch_accuracies = []
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= num_epochs, eta_min=0)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0 
        
        # Create a tqdm progress bar for the training process
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # Iterate through the batches of the training dataset
        for batch_idx, (inputs, targets) in progress_bar:
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the epoch and calculate the
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100  # Convert to percentage
        
        # Store the loss and accuracy
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        schedular.step()
        
        # Print statistics for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    print('Training completed!')
    
    # Return the losses and accuracies
    return epoch_losses, epoch_accuracies

